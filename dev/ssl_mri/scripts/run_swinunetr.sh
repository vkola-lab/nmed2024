#!/bin/bash

conda activate adrd

ps=16
vs=128
bs=8
heads=6
embed_dim=384
n_samples=1000
dataset="NACC_raw_${n_samples}"
outdim=8192
# arch="vit_tiny"
# export LD_PRELOAD=tcmalloc.so:$LD_PRELOAD
data_path="SET/YOUR/DATA/PATH"

#CUDA_VISIBLE_DEVICES=0
OMP_NUM_THREADS=1 NCCL_DEBUG=INFO python -m torch.distributed.run --nproc_per_node=1 --nnodes=1 --master_port=29759 \
            main_swinunetr.py --logdir ${ckptdir} --epochs 100 --num_steps=100000 --data_path ${data_path} --batch_size ${bs} --num_workers 3 \
            --use_checkpoint --eval_num 100