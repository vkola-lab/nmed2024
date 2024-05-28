#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml

# Request 4 CPUs
#$ -pe omp 4

#$ -m ea

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_memory=48G

#$ -l h_rt=12:00:00

conda activate py3.11
# module load python3/3.10.12
# module load pytorch/1.13.1

arch="ViTAutoEnc"
ps=32
vs=128
bs=2
num_heads=6
dataset="NACC_raw"

# arch="vit_tiny"
# export LD_PRELOAD=tcmalloc.so:$LD_PRELOAD
# export LD_LIBRARY_PATH=/share/pkg.7/miniconda/4.9.2/install/lib/

prefix="/projectnb/ivc-ml/dlteif/adrd_tool"
# data_path="${prefix}/data/training_cohorts/merged_data_nacc_nifd_stanford_adni_aibl_ppmi_revised_labels.csv"
train_path="${prefix}/data/training_cohorts/train_vld_test_split_updated/merged_train.csv"
vld_path="${prefix}/data/training_cohorts/train_vld_test_split_updated/merged_vld.csv"
test_path="${prefix}/data/training_cohorts/train_vld_test_split_updated/nacc_test_with_np_cli.csv"

# path="checkpoints/DINO_NACC_raw_ViTAutoEnc_voxel_size128patch_size32_batch_size4"
path="runs/SwinUNETR_pretrained_pixdim_minmaxnormalize"
ckpt="${path}/model_bestValMSE.pt"

# CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=3 
# CUDA_VISIBLE_DEVICES=2 
python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --master_port=29509 \
    train_cls.py --arch SwinUNETR --train_path ${train_path} --val_path ${vld_path} --test_path ${test_path} \
    --dataset ${dataset} --batch_size ${bs} --epochs 200 --patch_size ${ps} --image_size 128 --num_heads ${num_heads} --gpu 0 \
    --logdir_path checkpoints/SwinUNETR128_finetune_pretrained_cls_NC_MCI_DE_2layers_bs2 \
    --output checkpoints/SwinUNETR128_finetune_pretrained_cls_NC_MCI_DE_2layers_bs2 \
    --use_fp16 true --cls_layers 2 --finetune #--resume #--evaluate #--finetune #--resume 