# From https://github.com/xmuyzz/3D-CNN-PyTorch/blob/master/models/C3DNet.py

import torch
import torch.nn as nn
import sys
# from icecream import ic
import math

class C3D(torch.nn.Module):
    
    def __init__(self, tgt_modalities, in_channels=1, load_from_ckpt=None):
        
        super(C3D, self).__init__()
        self.conv_group1 = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.conv_group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.conv_group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            )
        self.conv_group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
            )

        # last_duration = int(math.floor(128 / 16))
        # last_size = int(math.ceil(128 / 32))
        self.fc1 = nn.Sequential(
            nn.Linear((512 * 15 * 9 * 9) , 512),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5))
        # self.fc = nn.Sequential(
        #     nn.Linear(4096, num_classes))
        
        self.fc = torch.nn.ModuleDict()
        for k in tgt_modalities:
            self.fc[k] = torch.nn.Linear(256, 1)
            
    def forward(self, x):
        # for k in x.keys():
        #     x[k] = x[k].to(torch.float32)
        
        # x = torch.stack([o for o in x.values()], dim=0)[0]
        # print(x.shape)
        
        out = self.conv_group1(x)
        out = self.conv_group2(out)
        out = self.conv_group3(out)
        out = self.conv_group4(out)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        # out = self.fc(out)
        
        tgt_iter = self.fc.keys()
        out_tgt = {k: self.fc[k](out).squeeze(1) for k in tgt_iter}
        return out_tgt
    

if __name__ == "__main__":
    model = C3D(tgt_modalities=['NC', 'MCI', 'DE'])
    print(model)
    x = torch.rand((1, 1, 128, 128, 128))
    # layers = list(model.features.named_children())
    # features = nn.Sequential(*list(model.features.children()))(x)
    # print(features.shape)
    print(sum(p.numel() for p in model.parameters()))
    # layer_found = False
    # features = None
    # desired_layer_name = 'transition3'

    # for name, layer in layers:
    #     if name == desired_layer_name:
    #         x = layer(x)
    #         print(x)
    # model(x)
    # print(features)