from .DeIT import *
from .dino import *
from .vision_transformer import *
from .embed_layer_3d_modality import VoxelEmbed, VoxelEmbed_Hybrid_no_average, \
                                        VoxelEmbed_Hybrid, VoxelNaiveProjection, \
                                        VoxelEmbed_Hybrid_no_average, VoxelEmbed_no_average
from .vit_3d_2d_pretrain import Feature3D_ViT2D_V2
from .vitautoenc import ViTAutoEnc

VALID_EMBED_LAYER={
    'VoxelEmbed': VoxelEmbed,
    'VoxelEmbed_no_zdim': VoxelNaiveProjection,
    'VoxelEmbed_no_average': VoxelEmbed_no_average,
    'VoxelEmbed': VoxelEmbed,
}

BACKBONE_EMBED_DIM={
    'deit_base_patch16_224': 768,
    'deit_small_patch16_224': 384,
    'deit_tiny_patch16_224': 192
}