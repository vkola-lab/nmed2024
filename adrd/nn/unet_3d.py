import sys
sys.path.append('..')
# from feature_extractor.for_image_data.backbone import CNN_GAP, ResNet3D, UNet3D
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
# from . import UNet3D
from .unet import UNet3D
from icecream import ic


class UNet3DBase(nn.Module):
    def __init__(self, n_class=1, act='relu', attention=False, pretrained=False, drop_rate=0.1, blocks=4):
        super(UNet3DBase, self).__init__()
        model = UNet3D(n_class=n_class, attention=attention, pretrained=pretrained, blocks=blocks)

        self.blocks = blocks

        self.down_tr64 = model.down_tr64
        self.down_tr128 = model.down_tr128
        self.down_tr256 = model.down_tr256
        self.down_tr512 = model.down_tr512
        if self.blocks == 5:
            self.down_tr1024 = model.down_tr1024
        # self.block_modules = nn.ModuleList([self.down_tr64, self.down_tr128, self.down_tr256, self.down_tr512])

        self.in_features = model.in_features
        # ic(attention)
        if attention:
            self.attention_module = model.attention_module
        #     self.attention_module = AttentionModule(512, n_class, drop_rate=drop_rate)
        # self.avgpool = nn.AvgPool3d((6,7,6), stride=(6,6,6))

    def forward(self, x, stage='normal', attention=False):
        # ic('UNet3DBase forward')
        self.out64, self.skip_out64 = self.down_tr64(x)
        # ic(self.out64.shape, self.skip_out64.shape)
        self.out128,self.skip_out128 = self.down_tr128(self.out64)
        # ic(self.out128.shape, self.skip_out128.shape)
        self.out256,self.skip_out256 = self.down_tr256(self.out128)
        # ic(self.out256.shape, self.skip_out256.shape)
        self.out512,self.skip_out512 = self.down_tr512(self.out256)
        # ic(self.out512.shape, self.skip_out512.shape)
        if self.blocks == 5:
            self.out1024,self.skip_out1024 = self.down_tr1024(self.out512)
        # ic(self.out1024.shape, self.skip_out1024.shape)
        # ic(hasattr(self, 'attention_module'))
        if hasattr(self, 'attention_module'):
            att, feats = self.attention_module(self.out1024 if self.blocks == 5 else self.out512)
        else:
            feats = self.out1024 if self.blocks == 5 else self.out512
        # ic(feats.shape)
        if attention:
            return att, feats
        return feats

        # self.out_up_256 = self.up_tr256(self.out512,self.skip_out256)
        # self.out_up_128 = self.up_tr128(self.out_up_256, self.skip_out128)
        # self.out_up_64 = self.up_tr64(self.out_up_128, self.skip_out64)
        # self.out = self.out_tr(self.out_up_64)

        # return self.out