#TODO: Need a better way to import models
from os import W_OK
import sys
from collections import OrderedDict

import torch
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, to_2tuple
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from .vision_transformer import VisionTransformer
from torch import nn
from einops import rearrange, repeat
from functools import partial
from .DeIT import *

def fit_dict(input_dict):
    """Fix name mismatch
    remove transformers
    proj_q, proj_k, proj_v  qkv
    """
    model_dict= OrderedDict()
    for k,v in input_dict.items():
        if 'pwff' in k:
            k=k.replace('pwff','mlp')
        if 'transformer' in k:
            model_dict[k[12:]] = v
        else:
            model_dict[k]=v
    for i in range(12):
        for s in ['weight','bias']:
            q = model_dict.pop('blocks.{}.attn.proj_q.{}'.format(i,s))
            k = model_dict.pop('blocks.{}.attn.proj_k.{}'.format(i,s))
            v = model_dict.pop('blocks.{}.attn.proj_v.{}'.format(i,s))
            model_dict['blocks.{}.attn.qkv.{}'.format(i,s)] = torch.cat((q,k,v),dim=0)

    return model_dict


class AMSoftmaxLayer(nn.Module):
    """AMSoftmaxLayer"""
    def __init__(self,
                 in_feats,
                 n_classes,
                 s=30.):
        super(AMSoftmaxLayer, self).__init__()
        self.s = s
        self.in_feats = in_feats
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes), requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)
    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm) * self.s
        return costh

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class FeatureVoxel_2DViT(nn.Module):
    __valid_model = {
    'deit_tiny_patch16_224': deit_tiny_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    'deit_base_patch16_224': deit_base_patch16_224,
    'deit_tiny_distilled_patch16_224': deit_tiny_distilled_patch16_224,
    'deit_small_distilled_patch16_224': deit_small_distilled_patch16_224,
    'deit_base_distilled_patch16_224': deit_base_distilled_patch16_224
    #'deit_base_patch16_384': deit_base_patch16_384,
    #'deit_base_distilled_patch16_384': deit_base_distilled_patch16_384,
    }
    def __init__(self, n_classes=10, input_shape=(32, 32, 32), transformer_backbone='deit_base_patch16_224'):
        super(FeatureVoxel_2DViT, self).__init__()
        self.n_classes = n_classes
        self.input_shape = input_shape
        if input_shape[0]==32:
            self.feat = torch.nn.Sequential(OrderedDict([
                ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                            out_channels=32, kernel_size=5, stride=2)),
                ('relu1', torch.nn.ReLU()),
                ('drop1', torch.nn.Dropout(p=0.2)),
                ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
                ('relu2', torch.nn.ReLU()),
                ('pool2', torch.nn.MaxPool3d(2)),
                ('drop2', torch.nn.Dropout(p=0.3))
            ]))
        elif input_shape[0]==128:
            self.feat = torch.nn.Sequential(OrderedDict([
                ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                            out_channels=8, kernel_size=5, stride=2)),
                ('relu1', torch.nn.ReLU()),
                ('drop1', torch.nn.Dropout(p=0.2)),
                ('conv3d_2', torch.nn.Conv3d(in_channels=8, out_channels=16, kernel_size=3)),
                ('relu2', torch.nn.ReLU()),
                ('pool2', torch.nn.MaxPool3d(2)),
                ('drop2', torch.nn.Dropout(p=0.3)),
                ('conv3d_3', torch.nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3)),
                ('relu3', torch.nn.ReLU()),
                ('pool3', torch.nn.MaxPool3d(2)),
                ('drop3', torch.nn.Dropout(p=0.3)),
                ('conv3d_4', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
                ('relu4', torch.nn.ReLU()),
                ('pool4', torch.nn.MaxPool3d(2)),
                ('drop4', torch.nn.Dropout(p=0.3)),
            ]))
        x = self.feat(torch.autograd.Variable(torch.rand((1, 1) + input_shape)))
        print(x.shape)
        dim_feat = 1
        for n in x.size()[2:]:
            dim_feat *= n
        self.dim_feat = dim_feat

        # this might be a naive way
        self.feature_connector = torch.nn.Sequential(OrderedDict([
           ('fc1', torch.nn.Linear(dim_feat, 196)),
           ('bn1', torch.nn.BatchNorm1d(32)),
           ('relu1', torch.nn.ReLU())
           # Layer of deconvolution
        ]))
        x = x.view(x.size(0), x.size(1), -1)
        x = self.feature_connector(x)
        print(x.shape)

        # make it 224 by 224 by 3
        self.up_scaling_2d = torch.nn.Sequential(OrderedDict([
           ('deconv1', Up(32, 16, bilinear=True)),
           ('deconv2', Up(16, 8, bilinear=True)),
           ('deconv3', Up(8, 4, bilinear=True)),
           ('deconv4', Up(4, 3, bilinear=False))
        ]))
        x = x.view(x.size(0), x.size(1), 14, 14)
        x = self.up_scaling_2d(x)
        print(x.shape)
        # One layer transformer
        if transformer_backbone not in self.__valid_model:
            raise ValueError("Unknown transformer backbone name!")

        self.transformer_backbone = transformer_backbone
        self.transformer= self.__valid_model[transformer_backbone](pretrained=True)
        # Freeze layers
        #for name, param in self.transformer.named_parameters():
        #    param.requires_grad = False

        # remove last layer
        self.transformer.head = nn.Linear(self.transformer.embed_dim, self.n_classes)
        x= self.transformer(x)

        if self.transformer_backbone in ['deit_tiny_distilled_patch16_224','deit_small_distilled_patch16_224','deit_base_distilled_patch16_224']:
            print(x[0].shape)
        else:
            print(x.shape)

    def forward(self, x):
        x = self.feat(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.feature_connector(x)
        x = x.view(x.size(0), x.size(1), 14, 14)
        x = self.up_scaling_2d(x)
        x = self.transformer(x)
        #TODO: Introduce distilled training correctly
        if self.transformer_backbone in ['deit_tiny_distilled_patch16_224','deit_small_distilled_patch16_224','deit_base_distilled_patch16_224']:
            return x[0]
        else:
            return x

class FeatureVoxel_2DViT_2layerhead(FeatureVoxel_2DViT):
    def __init__(self, n_classes=10, input_shape=(32, 32, 32), transformer_backbone='deit_base_patch16_224'):
        super(FeatureVoxel_2DViT_2layerhead, self).__init__(n_classes=n_classes,input_shape=input_shape,transformer_backbone=transformer_backbone)
        self.transformer.head = torch.nn.Sequential(OrderedDict([
            ('fc1', torch.nn.Linear(self.transformer.embed_dim, 256)),
            ('relu1', torch.nn.ReLU()),
            ('drop3', torch.nn.Dropout(p=0.3)),
            ('fc2', torch.nn.Linear(256, self.n_classes))
        ]))





class Feature3D_ViT2D(nn.Module):
    '''
        3D Feature
    '''
    __valid_model = {
    'deit_tiny_patch16_224': deit_tiny_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    'deit_base_patch16_224': deit_base_patch16_224,
    #'deit_tiny_distilled_patch16_224': deit_tiny_distilled_patch16_224,
    #'deit_small_distilled_patch16_224': deit_small_distilled_patch16_224,
    #'deit_base_distilled_patch16_224': deit_base_distilled_patch16_224
    #'deit_base_patch16_384': deit_base_patch16_384,
    #'deit_base_distilled_patch16_384': deit_base_distilled_patch16_384,
    }

    def __init__(self, embed_layer=None, n_classes=10, data_shape=None, transformer_backbone='deit_base_patch16_224', pretrained=True, pos_embedding=None):
        super().__init__()

        if transformer_backbone not in self.__valid_model:
            raise ValueError("Unknown transformer backbone name!")

        self.n_classes = n_classes

        self.transformer_backbone = transformer_backbone
        self.transformer= self.__valid_model[transformer_backbone](pretrained=pretrained)

        # replace patch_embed layer
        self.transformer.patch_embed = embed_layer(embed_dim=self.transformer.embed_dim)
        # change positional encoding
        self.transformer.num_patches = self.transformer.patch_embed.num_patches
        if pos_embedding is None or pos_embedding=="default":
            self.transformer.pos_embed = nn.Parameter(torch.zeros(1, self.transformer.patch_embed.num_patches + 1, self.transformer.embed_dim))
            trunc_normal_(self.transformer.pos_embed, std=.02)
        elif pos_embedding=="no_embed":
            self.transformer.pos_embed =  nn.Parameter(torch.zeros(1, self.transformer.patch_embed.num_patches + 1, self.transformer.embed_dim, requires_grad=False))
            self.pos_drop = nn.Identity()
        elif pos_embedding=="group_embed":
            self.transformer.pos_embed = nn.Parameter(torch.zeros(1, self.transformer.patch_embed.num_patches + 1, self.transformer.embed_dim))
            trunc_normal_(self.transformer.pos_embed, std=.02)
            self.transformer.forward_features = self.__group_embedding_forward_features
        elif pos_embedding=="weight_sharing":
            self.transformer.pos_embed = nn.Parameter(torch.zeros(1, self.transformer.patch_embed.num_patches + 1, self.transformer.embed_dim))
            trunc_normal_(self.transformer.pos_embed, std=.02)
            self.transformer.forward_features = self.__group_embedding_forward_features
        else:
            raise ValueError
        # replace last head layer
        self.transformer.head = nn.Linear(self.transformer.embed_dim, self.n_classes)

    def __group_embedding_forward_features(self, x):
        pass

    def __weight_sharing_forward_features(self, x):
        pass

    def forward(self,x):
        return self.transformer(x)


class Feature3D_ViT2D_V2(VisionTransformer):
    '''
        3D Feature
    '''
    __valid_model = {
            'deit_tiny_patch16_224': {
                'patch_size':16,
                'embed_dim':192,
                'depth':12,
                'num_heads':3,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            'deit_small_patch16_224': {
                'patch_size':16,
                'embed_dim':384,
                'depth':12,
                'num_heads':6,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            'deit_base_patch16_224': {
                'patch_size':16,
                'embed_dim':768,
                'depth':12,
                'num_heads':3,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            'deit_base_distilled_patch16_224': {
                'patch_size':16,
                'embed_dim':768,
                'depth':12,
                'num_heads':3,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            'vit_base_patch16_224_21k': {
                'patch_size':16,
                'embed_dim':768,
                'depth':12,
                'num_heads':3,
                'mlp_ratio':4,
                'qkv_bias':True,
                'norm_layer':partial(nn.LayerNorm, eps=1e-6)
            },
            }
    __valid_model_pretrain_dict_url = {
        'deit_tiny_patch16_224': "https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
        'deit_small_patch16_224': "https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
        'deit_base_patch16_224': "https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
        'deit_base_distilled_patch16_224': "https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth",
        'vit_base_patch16_224_21k': "./3rd_party/B_16.pth"
    }

    def __init__(self, n_classes=10, embed_layer=None, data_shape=None, transformer_backbone='deit_base_patch16_224', pretrained=True, pos_embedding=None, **kwargs):
        self.transformer_backbone = transformer_backbone
        self.pretrained = pretrained
        transformer_dict = self.__load_transformer_config()
        super().__init__(
            patch_size= transformer_dict['patch_size'],
            embed_dim= transformer_dict['embed_dim'],
            depth= transformer_dict['depth'],
            num_heads= transformer_dict['num_heads'],
            mlp_ratio= transformer_dict['mlp_ratio'],
            qkv_bias= transformer_dict['qkv_bias'],
            norm_layer= transformer_dict['norm_layer'],
        )
        self.default_cfg = _cfg()
        self.url = self.__valid_model_pretrain_dict_url[self.transformer_backbone]
        self.dist_token = None

        self.n_classes = n_classes

        # load weight
        ic(self.transformer_backbone)
        self.__load_backbone_weight()

        # change patch embedding layer
        self.voxel_embed = embed_layer
        ic(self.voxel_embed)

        # replace last head layer
        if kwargs.get('head')=='AMSoftmax':
            self.voxel_head = AMSoftmaxLayer(self.embed_dim, self.n_classes)
        else:
            self.voxel_head = nn.Linear(self.embed_dim, self.n_classes)

        # Setup different positional embedding
        self.pos_embed_type = pos_embedding

        if pos_embedding is None or pos_embedding=="default":
            self.voxel_pos_embed = nn.Parameter(torch.zeros(1, self.voxel_embed.num_patches + 1, self.embed_dim))
            trunc_normal_(self.pos_embed, std=.02)
            ic(self.voxel_pos_embed.size())
        elif pos_embedding=="no_embed":
            if self.patch_embed.num_patches != 196:
                self.voxel_pos_embed = nn.Parameter(torch.zeros(1, self.voxel_embed.num_patches + 1, self.embed_dim, requires_grad=False))
                trunc_normal_(self.pos_embed, std=.02)
            self.pos_drop = nn.Identity()
        elif pos_embedding=="group_embed":
            if self.patch_embed.patch_size != 14:
                self.voxel_pos_embed = nn.Parameter(torch.zeros(1, self.voxel_embed.patch_size**2 + 1, self.embed_dim))
                trunc_normal_(self.pos_embed, std=.02)
            self.group_embed = nn.TransformerEncoderLayer(d_model = self.embed_dim, dim_feedforward= self.embed_dim, nhead = 4)
            self.group_pos_embed = nn.Parameter(torch.zeros(1, self.voxel_embed.patch_size + 1, self.embed_dim))
            self.group_cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        elif pos_embedding=="weight_sharing":
            if self.patch_embed.patch_size != 14:
                self.voxel_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.patch_size**2 + 1, self.embed_dim))
                trunc_normal_(self.pos_embed, std=.02)
        else:
            raise ValueError("Unknown positional embedding scheme!")

    def __load_transformer_config(self):

        if self.transformer_backbone not in self.__valid_model:
            raise ValueError("Unknown transformer backbone name!")

        return self.__valid_model[self.transformer_backbone]


    def __load_backbone_weight(self):
        if self.pretrained:
            print(self.url)
            if '21k' in self.transformer_backbone:
                pretrained_dict = torch.load(self.url, map_location="cpu")
                pretrained_dict = fit_dict(pretrained_dict)
            else:
                checkpoint = torch.hub.load_state_dict_from_url(
                    url=self.url,
                    map_location="cpu", check_hash=True
                    )
                pretrained_dict = checkpoint["model"]
            # load partial dict_file (except for pos_embed and last layer)
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # key_list=  list(pretrained_dict.keys())
            # for k in key_list:
            #     if 'blocks.0.attn' in k:
            #         del pretrained_dict[k]

            #print(pretrained_dict.keys())
            #model_dict.update(pretrained_dict)
            if 'distilled' in self.transformer_backbone:
                pretrained_dict['pos_embed']=pretrained_dict['pos_embed'][:,1:,:]
            


            self.load_state_dict(pretrained_dict, strict=False)

            if not isinstance(self.head, nn.Identity):
                self.head.weight.requires_grad = False
                self.head.bias.requires_grad = False
            self.pos_embed.requires_grad = False
            for param in self.patch_embed.parameters():
                param.requires_grad = False


    def forward_images(self, x):
        #TODO: Fix Gradient of patch_embed and pos_embed
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)


        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])

    def forward_features(self, x):
        pos_embedding = self.pos_embed_type
        ic(pos_embedding)
        if pos_embedding is None or pos_embedding=="default" or pos_embedding=="no_embed":
            ic(x.size())
            ic(self.voxel_embed.voxel_size)
            x = self.voxel_embed(x)
            ic(x.size())
            x = x.flatten(2).transpose(1,2)
            ic(x.size())
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            ic(cls_token.size())
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
                ic(x.size())
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
                ic(self.dist_token)

            # Difference starting from here:
            ic(self.voxel_pos_embed.size(), x.size())
            x = self.pos_drop(x + self.voxel_pos_embed)
            ic(x.size())
            for blk in self.blocks:
                x = blk(x)
                # ic(x.size())
            x = self.norm(x)
            ic(x.size(), x[:,0].size())
            return x[:, 0]

        elif pos_embedding=="group_embed":
            x = self.voxel_embed(x)
            ic(x.size())
            x = rearrange(x, 'b c px py pz -> (b px py) pz c')
            ic(x.size())

            group_cls_token = self.group_cls_token.expand(x.shape[0], -1, -1)
            ic(group_cls_token.size())
            x = torch.cat((group_cls_token, x), dim=1)
            ic(x.size())
            x = self.pos_drop(x + self.group_pos_embed)
            ic(x.size())
            x = self.group_embed(x)
            ic(x.size())

            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            x = x[:, 0]
            x = rearrange(x, '(b px py) c -> b (px py) c', px = self.voxel_embed.patch_size, py = self.voxel_embed.patch_size)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

            x = self.pos_drop(x + self.voxel_pos_embed)
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x[:, 0]

        elif pos_embedding=="weight_sharing":
            # average sharing
            x = self.voxel_embed(x)
            x = rearrange(x, 'b c px py pz -> b (px py) c pz')
            avg = torch.zeros((x.shape[0], x.shape[2])).cuda()
            for i in range(x.shape[-1]):
                z = x[:,:,:, i]
                cls_token = self.cls_token.expand(z.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                if self.dist_token is None:
                    z = torch.cat((cls_token, z), dim=1)
                else:
                    z = torch.cat((cls_token, self.dist_token.expand(z.shape[0], -1, -1), z), dim=1)

                # Difference starting from here:

                z = self.pos_drop(z + self.voxel_pos_embed)
                for blk in self.blocks:
                    z = blk(z)
                z = self.norm(z)
                avg = avg + z[:, 0]
            avg = avg/ x.shape[-1]
            return avg
        else:
            raise ValueError("Unknown positional embedding scheme!")

    def get_last_selfattention(self, x):
        pos_embedding = self.pos_embed_type
        ic(pos_embedding)
        if pos_embedding is None or pos_embedding=="default" or pos_embedding=="no_embed":
            ic(x.size())
            ic(self.voxel_embed.voxel_size)
            x = self.voxel_embed(x)
            ic(x.size())
            x = x.flatten(2).transpose(1,2)
            ic(x.size())
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            ic(cls_token.size())
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
                ic(x.size())
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
                ic(self.dist_token)

            # Difference starting from here:
            ic(self.voxel_pos_embed.size(), x.size())
            x = self.pos_drop(x + self.voxel_pos_embed)
            ic(x.size())
            for i, blk in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    x = blk(x)
                else:
                    return blk(x, return_attention=True)
                # ic(x.size())
           

        elif pos_embedding=="group_embed":
            x = self.voxel_embed(x)
            x = rearrange(x, 'b c px py pz -> (b px py) pz c')

            group_cls_token = self.group_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((group_cls_token, x), dim=1)
            x = self.pos_drop(x + self.group_pos_embed)
            x = self.group_embed(x)

            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            x = x[:, 0]
            x = rearrange(x, '(b px py) c -> b (px py) c', px = self.voxel_embed.patch_size, py = self.voxel_embed.patch_size)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
            if self.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

            x = self.pos_drop(x + self.voxel_pos_embed)
            for i, blk in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    x = blk(x)
                else:
                    return blk(x, return_attention=True)
            

        elif pos_embedding=="weight_sharing":
            # average sharing
            x = self.voxel_embed(x)
            x = rearrange(x, 'b c px py pz -> b (px py) c pz')
            avg = torch.zeros((x.shape[0], x.shape[2])).cuda()
            for i in range(x.shape[-1]):
                z = x[:,:,:, i]
                cls_token = self.cls_token.expand(z.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
                if self.dist_token is None:
                    z = torch.cat((cls_token, z), dim=1)
                else:
                    z = torch.cat((cls_token, self.dist_token.expand(z.shape[0], -1, -1), z), dim=1)

            pass
        else:
            raise ValueError("Unknown positional embedding scheme!")
    
    def forward(self,x, is_training=True):
        # ic.enable()
        x = self.forward_features(x)
        if not is_training:
            x = self.voxel_head(x)
        return x





if __name__ =="__main__":

    # a =  FeatureVoxel_2DViT(input_shape=(128, 128, 128))
    # for name, param in a.named_parameters():
    #     if 'transformer' in name:
    #         print(name, param.requires_grad)
    a = VoxelEmbed()
