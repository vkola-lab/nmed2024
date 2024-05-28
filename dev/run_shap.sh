#!/bin/bash -l

# Set SCC project
#$ -P ivc-ml

# Request 4 CPUs
#$ -pe omp 3

#$ -m ea

# Request 1 GPU 
#$ -l gpus=1
#$ -l gpu_memory=48G

#$ -l h_rt=48:00:00


# run this script from adrd_tool/

conda activate py3.11
# conda activate adrd
pip install -e .

# CUDA_VISIBLE_DEVICES=1 
python dev/backbone_shap.py