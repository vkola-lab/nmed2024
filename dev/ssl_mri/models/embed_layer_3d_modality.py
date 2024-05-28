
'''
    TODO: Put all 3D embed layer here
'''
import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict

class VoxelEmbed(nn.Module):
    """ Voxel to Patch Embedding (Simplest 3D CNN)
    """
    def __init__(self, voxel_size=128, cell_size=16, patch_size=8, in_chans=1, embed_dim=768):
        super().__init__()
        self.voxel_size = (voxel_size,voxel_size,voxel_size)
        self.cell_size = (cell_size, cell_size,cell_size)
        self.patch_size = patch_size
        num_patches = (voxel_size // cell_size) ** 2
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = torch.nn.Sequential(OrderedDict([
                ('conv3d_1', torch.nn.Conv3d(in_channels=in_chans,
                                            out_channels=embed_dim, kernel_size=cell_size, stride=cell_size)),
                #('relu1', torch.nn.ReLU()),
                #('pool1', torch.nn.MaxPool3d(2)),
                #('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=embed_dim, kernel_size=3))
            ]))
        # [batch_size, embed_dim, 14, 14, 14]
        # x = self.proj(torch.autograd.Variable(torch.rand((1, 1) + self.voxel_size)))
        # print(x.shape)

    def forward(self, x):
        B, C, H, W, V = x.shape
        # FIXME look at relaxing size constraints
        ic(x.size())
        # assert H == self.voxel_size[0] and W == self.voxel_size[1] and V == self.voxel_size[2], \
        #     f"Input voxel size ({H}*{W}*{V}) doesn't match model ({self.voxel_size[0]}*{self.voxel_size[1]}*{self.voxel_size[2]})."
        x = self.proj(x)
        ic(x.size())
        x = torch.mean(x,dim=4)
        ic(x.size())
        #x = torch.mean(self.proj(x),dim=4).flatten(2).transpose(1, 2)
        return x

class VoxelEmbed_no_average(nn.Module):
    def __init__(self, voxel_size=128, cell_size=16, patch_size=8, in_chans=1, embed_dim=768):
        super().__init__()
        self.voxel_size = (voxel_size,voxel_size,voxel_size)
        self.cell_size = (cell_size, cell_size,cell_size)
        self.patch_size = patch_size
        num_patches = (voxel_size // cell_size) ** 2
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = torch.nn.Sequential(OrderedDict([
                ('conv3d_1', torch.nn.Conv3d(in_channels=in_chans,
                                            out_channels=embed_dim, kernel_size=cell_size, stride=cell_size)),
                #('relu1', torch.nn.ReLU()),
                #('pool1', torch.nn.MaxPool3d(2)),
                #('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=embed_dim, kernel_size=3))
            ]))
        # [batch_size, embed_dim, 6, 6, 6]
        # x = self.proj(torch.autograd.Variable(torch.rand((1, 1) + self.voxel_size)))
        # print(x.shape)

    def forward(self, x):
        B, C, H, W, V = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.voxel_size[0] and W == self.voxel_size[1] and V == self.voxel_size[2], \
            f"Input voxel size ({H}*{W}*{V}) doesn't match model ({self.voxel_size[0]}*{self.voxel_size[1]}*{self.voxel_size[2]})."
        #x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class VoxelEmbed_Hybrid(nn.Module):
    """ Voxel to Patch Embedding (Simplest 3D CNN)
    """
    def __init__(self, voxel_size=128, cell_size=1, patch_size=1, in_chans=1, embed_dim=768):
        super().__init__()
        self.voxel_size = (voxel_size,voxel_size,voxel_size)
        self.cell_size = (cell_size, cell_size,cell_size)
        self.patch_size = patch_size
        num_patches = 36
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.backbone = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))
        self.proj = torch.nn.Conv3d(in_channels=32, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        

    def forward(self, x):
        B, C, H, W, V = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.voxel_size[0] and W == self.voxel_size[1] and V == self.voxel_size[2], \
            f"Input voxel size ({H}*{W}*{V}) doesn't match model ({self.voxel_size[0]}*{self.voxel_size[1]}*{self.voxel_size[2]})."
        if self.voxel_size[0]==128:
            x = torch.nn.functional.interpolate(x, size=(32,32,32), mode="trilinear")

        x = self.backbone(x) 
        #x = torch.mean(self.proj(x),dim=4).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class VoxelEmbed_Hybrid_no_average(nn.Module):
    def __init__(self, voxel_size=128, cell_size=1, patch_size=1, in_chans=1, embed_dim=768):
        super().__init__()
        self.voxel_size = (voxel_size,voxel_size,voxel_size)
        self.cell_size = (cell_size, cell_size,cell_size)
        self.patch_size = patch_size
        num_patches = 216
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.backbone = torch.nn.Sequential(OrderedDict([
            ('conv3d_1', torch.nn.Conv3d(in_channels=1,
                                         out_channels=32, kernel_size=5, stride=2)),
            ('relu1', torch.nn.ReLU()),
            ('drop1', torch.nn.Dropout(p=0.2)),
            ('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3)),
            ('relu2', torch.nn.ReLU()),
            ('pool2', torch.nn.MaxPool3d(2)),
            ('drop2', torch.nn.Dropout(p=0.3))
        ]))
        self.proj = torch.nn.Conv3d(in_channels=32, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        

    def forward(self, x):
        B, C, H, W, V = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.voxel_size[0] and W == self.voxel_size[1] and V == self.voxel_size[2], \
            f"Input voxel size ({H}*{W}*{V}) doesn't match model ({self.voxel_size[0]}*{self.voxel_size[1]}*{self.voxel_size[2]})."

        if self.voxel_size[0]==128:
            x = torch.nn.functional.interpolate(x, size=(32,32,32), mode="trilinear")
        x = self.backbone(x)
        #x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


# class VoxelEmbed(nn.Module):
#     """ Voxel to Patch Embedding (Simplest 3D CNN)
#     """
#     def __init__(self, voxel_size=128, cell_size=16, patch_size=8, in_chans=1, embed_dim=768):
#         super().__init__()
#         self.voxel_size = (voxel_size,voxel_size,voxel_size)
#         self.cell_size = (cell_size, cell_size,cell_size)
#         self.patch_size = patch_size
#         num_patches = (voxel_size // patch_size) ** 2
#         # num_patches = patch_size ** 2
#         self.num_patches = num_patches
#         self.embed_dim = embed_dim

#         self.proj = torch.nn.Sequential(OrderedDict([
#                 ('conv3d_1', torch.nn.Conv3d(in_channels=in_chans,
#                                             out_channels=embed_dim, kernel_size=cell_size, stride=cell_size)),
#                 #('relu1', torch.nn.ReLU()),
#                 #('pool1', torch.nn.MaxPool3d(2)),
#                 #('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=embed_dim, kernel_size=3))
#             ]))
#         # [batch_size, embed_dim, 14, 14, 14]
#         # x = self.proj(torch.autograd.Variable(torch.rand((1, 1) + self.voxel_size)))
#         # print(x.shape)

#     def forward(self, x):
#         B, C, H, W, V = x.shape
#         # FIXME look at relaxing size constraints
#         # assert H == self.voxel_size[0] and W == self.voxel_size[1] and V == self.voxel_size[2], \
#         #     f"Input voxel size ({H}*{W}*{V}) doesn't match model ({self.voxel_size[0]}*{self.voxel_size[1]}*{self.voxel_size[2]})."
#         x = torch.mean(self.proj(x),dim=4)
#         #x = torch.mean(self.proj(x),dim=4).flatten(2).transpose(1, 2)
#         return x

class VoxelNaiveProjection(nn.Module):
    """ Voxel to Patch Embedding (Simplest 3D CNN)
    """
    def __init__(self, voxel_size=128, cell_size=16, patch_size=8, in_chans=1, embed_dim=768):
        super().__init__()
        self.voxel_size = (voxel_size,voxel_size,voxel_size)
        self.cell_size = (cell_size, cell_size,cell_size)
        self.patch_size = patch_size
        num_patches = patch_size ** 2
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj = torch.nn.Sequential(OrderedDict([
                ('conv2d_1', torch.nn.Conv2d(in_channels=in_chans,
                                            out_channels=embed_dim, kernel_size=cell_size, stride=cell_size)),
                #('relu1', torch.nn.ReLU()),
                #('pool1', torch.nn.MaxPool3d(2)),
                #('conv3d_2', torch.nn.Conv3d(in_channels=32, out_channels=embed_dim, kernel_size=3))
            ]))
        # [batch_size, embed_dim, 14, 14, 14]
        # x = self.proj(torch.autograd.Variable(torch.rand((1, 1) + self.voxel_size)))
        # print(x.shape)

    def forward(self, x):
        B, C, H, W, V = x.shape
        # ic.enable()
        ic(x.size())
        # FIXME look at relaxing size constraints
        # assert H == self.voxel_size[0] and W == self.voxel_size[1] and V == self.voxel_size[2], \
            # f"Input voxel size ({H}*{W}*{V}) doesn't match model ({self.voxel_size[0]}*{self.voxel_size[1]}*{self.voxel_size[2]})."
        x = torch.clamp(torch.sum(x, dim=4),min=0,max=1)
        ic(x.size())
        x = self.proj(x)
        ic(x.size())
        return x
