#!/bin/bash

# run this script from adrd_tool/

conda activate adrd

# install the package
# cd adrd_tool
pip install .

# define the variables
prefix="/home/skowshik/ADRD_repo/pipeline_v1_main/adrd_tool"
data_path="${prefix}/data/train_vld_test_split_updated/merged_train.csv"
ckpt_path="/home/skowshik/publication_ADRD_repo/adrd_tool/dev/ckpt/ckpt_densenet_232_all_stripped_mni.pt"
# backend=C3D
backend=DenseNet
emb_type=ALL
# run train.py
python dev/imaging_train.py --data_path $data_path --ckpt_path $ckpt_path --num_epochs 256 --batch_size 64 --lr 1e-3 --gamma 2 --emb_type $emb_type --img_size "(182,218,182)" --backend $backend #--wandb_

