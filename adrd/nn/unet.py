import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import models
from torch.nn import init
import torch.nn.functional as F
from icecream import ic


class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):

        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(input.dim()))
        #super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, in_chan, out_chan, act):
        super(LUConv, self).__init__()
        self.conv1 = nn.Conv3d(in_chan, out_chan, kernel_size=3, padding=1)
        self.bn1 = ContBatchNorm3d(out_chan)

        if act == 'relu':
            self.activation = nn.ReLU(out_chan)
        elif act == 'prelu':
            self.activation = nn.PReLU(out_chan)
        elif act == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            raise

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        return out


def _make_nConv(in_channel, depth, act, double_chnnel=False):
    if double_chnnel:
        layer1 = LUConv(in_channel, 32 * (2 ** (depth+1)),act)
        layer2 = LUConv(32 * (2 ** (depth+1)), 32 * (2 ** (depth+1)),act)
    else:
        layer1 = LUConv(in_channel, 32*(2**depth),act)
        layer2 = LUConv(32*(2**depth), 32*(2**depth)*2,act)

    return nn.Sequential(layer1,layer2)


class DownTransition(nn.Module):
    def __init__(self, in_channel,depth, act):
        super(DownTransition, self).__init__()
        self.ops = _make_nConv(in_channel, depth,act)
        self.maxpool = nn.MaxPool3d(2)
        self.current_depth = depth

    def forward(self, x):
        if self.current_depth == 3:
            out = self.ops(x)
            out_before_pool = out
        else:
            out_before_pool = self.ops(x)
            out = self.maxpool(out_before_pool)
        return out, out_before_pool

class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, depth,act):
        super(UpTransition, self).__init__()
        self.depth = depth
        self.up_conv = nn.ConvTranspose3d(inChans, outChans, kernel_size=2, stride=2)
        self.ops = _make_nConv(inChans+ outChans//2,depth, act, double_chnnel=True)

    def forward(self, x, skip_x):
        out_up_conv = self.up_conv(x)
        concat = torch.cat((out_up_conv,skip_x),1)
        out = self.ops(concat)
        return out

class OutputTransition(nn.Module):
    def __init__(self, inChans, n_labels):

        super(OutputTransition, self).__init__()
        self.final_conv = nn.Conv3d(inChans, n_labels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.sigmoid(self.final_conv(x))
        return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate, kernel, pooling, BN=True, relu_type='leaky'):
        super().__init__()
        kernel_size, kernel_stride, kernel_padding = kernel
        pool_kernel, pool_stride, pool_padding = pooling
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, kernel_stride, kernel_padding, bias=False)
        self.pooling = nn.MaxPool3d(pool_kernel, pool_stride, pool_padding)
        self.BN = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(inplace=False) if relu_type=='leaky' else nn.ReLU(inplace=False)
        self.dropout = nn.Dropout(drop_rate, inplace=False) 
       
    def forward(self, x):
        x = self.conv(x)
        x = self.pooling(x)
        x = self.BN(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
class AttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.1):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.attention = ConvLayer(in_channels, out_channels, drop_rate, (1, 1, 0), (1, 1, 0))

    def forward(self, x, return_attention=True):
        feats = self.conv(x)
        att = F.softmax(self.attention(x))

        out = feats * att

        if return_attention:
            return att, out
        
        return out 

class UNet3D(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, n_class=1, act='relu', pretrained=False, input_size=(1,1,182,218,182), attention=False, drop_rate=0.1, blocks=4):
        super(UNet3D, self).__init__()

        self.blocks = blocks
        self.down_tr64 = DownTransition(1,0,act)
        self.down_tr128 = DownTransition(64,1,act)
        self.down_tr256 = DownTransition(128,2,act)
        self.down_tr512 = DownTransition(256,3,act)

        self.up_tr256 = UpTransition(512, 512,2,act)
        self.up_tr128 = UpTransition(256,256, 1,act)
        self.up_tr64 = UpTransition(128,128,0,act)
        self.out_tr = OutputTransition(64, 1)

        self.pretrained = pretrained
        self.attention = attention
        if pretrained:
            print("Using image pretrained model checkpoint")
            weight_dir = '/home/skowshik/ADRD_repo/img_pretrained_ckpt/Genesis_Chest_CT.pt'
            checkpoint = torch.load(weight_dir)
            state_dict = checkpoint['state_dict']
            unParalled_state_dict = {}
            for key in state_dict.keys():
                unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
            self.load_state_dict(unParalled_state_dict)
            del self.up_tr256
            del self.up_tr128
            del self.up_tr64
            del self.out_tr
        
        if self.blocks == 5:
            self.down_tr1024 = DownTransition(512,4,act)
            

        # self.conv1 = nn.Conv3d(512, 256, 1, 1, 0, bias=False)
        # self.conv2 = nn.Conv3d(256, 128, 1, 1, 0, bias=False)
        # self.conv3 = nn.Conv3d(128, 64, 1, 1, 0, bias=False)

        if attention:
            self.attention_module = AttentionModule(1024 if self.blocks==5 else 512, n_class, drop_rate=drop_rate)
        # Output.
        self.avgpool = nn.AvgPool3d((6,7,6), stride=(6,6,6))

        dummy_inp = torch.rand(input_size)
        dummy_feats = self.forward(dummy_inp, stage='get_features')
        dummy_feats = dummy_feats[0]
        self.in_features = list(dummy_feats.shape)
        ic(self.in_features)

        self._init_weights()

    def _init_weights(self):
        if not self.pretrained:
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    init.kaiming_normal_(m.weight)
                elif isinstance(m, ContBatchNorm3d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    init.kaiming_normal_(m.weight)
                    init.constant_(m.bias, 0)
        elif self.attention:
            for m in self.attention_module.modules():
                if isinstance(m, nn.Conv3d):
                    init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm3d):
                    init.constant_(m.weight, 1)
                    init.constant_(m.bias, 0)
        else:
            pass
        # Zero initialize the last batchnorm in each residual branch.
        # for m in self.modules():
        #     if isinstance(m, BottleneckBlock):
        #         init.constant_(m.out_conv.bn.weight, 0)
    
    def forward(self, x, stage='normal', attention=False):
        ic('backbone forward')
        self.out64, self.skip_out64 = self.down_tr64(x)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)
        if self.blocks == 5:
            self.out1024,self.skip_out1024 = self.down_tr1024(self.out512)
            ic(self.out1024.shape)
        # self.out = self.conv1(self.out512)
        # self.out = self.conv2(self.out)
        # self.out = self.conv3(self.out)
        # self.out = self.conv(self.out)
        ic(hasattr(self, 'attention_module'))
        if hasattr(self, 'attention_module'):
            att, feats = self.attention_module(self.out1024 if self.blocks==5 else self.out512)
        else:
            feats = self.out1024 if self.blocks==5 else self.out512
        ic(feats.shape)
        if attention:
            return att, feats
        return feats