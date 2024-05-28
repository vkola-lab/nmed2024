#!/bin/bash -l

# Set SCC project

# Request 4 CPUs
#$ -pe omp 4

#$ -m ea

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_memory=80G
#$ -l h_rt=12:00:00

conda activate adrd
module load python3/3.8.10
module load pytorch/1.13.1

data_path="/projectnb/ivc-ml/dlteif/NACC_raw"
path="/projectnb/ivc-ml/dlteif/pretrained_models"
#path="runs/SwinUNETR_pretrained_pixdim_minmaxnormalize"
ckpt="model_swinvit.pt"

CUDA_VISIBLE_DEVICES=1 python infer_swinunetr.py --data_dir=${data_path} --feature_size=48 --infer_overlap=0.3 --pretrained_dir ${path} --pretrained_model_name=${ckpt} --roi_x=128 --roi_y=128 --roi_z=128 --batch_size=1