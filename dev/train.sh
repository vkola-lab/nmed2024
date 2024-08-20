#!/bin/bash -l

# run this script from adrd_tool/

conda activate adrd
pip install .

# install the package
# cd adrd_tool
# pip install -e .

# define the variables
prefix="."
data_path="${prefix}/data/training_cohorts/new_nacc_revised_selection.csv"
train_path="${prefix}/data/train_vld_test_split_updated/demo_train.csv"
vld_path="${prefix}/data/train_vld_test_split_updated/demo_vld.csv"
test_path="${prefix}/data/train_vld_test_split_updated/nacc_test_with_np_cli.csv"
cnf_file="${prefix}/dev/data/toml_files/default_conf_new.toml"
imgnet_ckpt="${prefix}/dev/ckpt/ckpt_densenet_232_all_stripped_mni.pt"


# Note for setting the flags
# 1. If training without MRIs
# img_net="NonImg"
# img_mode = -1
# 2. if training with MRIs
# img_net: [ViTAutoEnc, DenseNet, SwinUNETR]
# img_mode = 0
# 3. if training with MRI embeddings
# img_net: [ViTEmb, DenseNetEMB, SwinUNETREMB, NonImg]
# img_mode = 1


# img_net="NonImg"
# img_mode=-1


img_net="SwinUNETREMB"
img_mode=1

# img_net="DenseNet"
# img_mode=0

ckpt_path="./dev/ckpt/ckpt_swinunetr_stripped_MNI.pt"

# run train.py 
python dev/train.py --data_path $data_path --train_path $train_path --vld_path $vld_path --test_path $test_path --cnf_file $cnf_file --ckpt_path $ckpt_path --d_model 256 --nhead 1 \
                    --num_epochs 256 --batch_size 128 --lr 0.001 --gamma 0 --img_mode $img_mode --img_net $img_net --img_size "(182,218,182)" --imgnet_ckpt ${imgnet_ckpt} \
                    --patch_size 16 --ckpt_path $ckpt_path --train_imgnet --cnf_file ${cnf_file} --train_path ${train_path} --vld_path ${vld_path} --data_path ${data_path}  \
                    --fusion_stage middle --imgnet_layers 4 --weight_decay 0.0005 --ranking_loss --save_intermediate_ckpts  --wandb #--train_imgnet #--load_from_ckpt #--balanced_sampling