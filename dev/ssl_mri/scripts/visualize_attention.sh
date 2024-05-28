#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml

# Request 4 CPUs
#$ -pe omp 3

#$ -m ea

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_memory=48G

#$ -l h_rt=1:00:00

module load python3/3.8.10
module load pytorch/1.13.1
conda activate adrd

arch="ViTAutoEnc"
ps=32
bs=8
path="checkpoints/DINO_NACC_raw_ViTAutoEnc_voxel_size128patch_size16_batch_size2"
ckpt="${path}/checkpoint0003.pth"
# mri="/projectnb/ivc-ml/dlteif/ayan_datasets/NACC_ALL/npy/mri8590_1.2.840.113619.2.408.5282380.4561270.27826.1525459928.930.npy"
mri="/projectnb/ivc-ml/dlteif/NACC_raw/FLAIR/mri5152ni/NACC815048/3D_CORO_T1_3/1.2.840.113619.2.25.1.1762870024.1255126062.766.nii"
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=5 python visualize_attention.py --arch $arch --embed_layer VoxelEmbed --patch_size $ps --pretrained_weights $ckpt --checkpoint_key student --image_path $mri --image_size 128 128 128 --output_dir $path/visualizations --threshold 0.1 --gpu 0 --num_heads 6