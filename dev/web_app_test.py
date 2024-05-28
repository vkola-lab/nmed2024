# %%
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc, confusion_matrix, \
     RocCurveDisplay, precision_score, recall_score, average_precision_score, PrecisionRecallDisplay, precision_recall_curve
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
# from adrd import data
# from data import CSVDataset
from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel, DynamicCalibratedClassifier, StaticCalibratedClassifier
from adrd.utils.misc import get_and_print_metrics_multitask
from adrd.utils.misc import get_metrics, print_metrics
import numpy as np
from tqdm import tqdm
import json
# from adrd.data import _conf
import adrd.utils.misc
import torch
import os
from icecream import ic
ic.disable()


from torchvision import transforms

import monai
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    CopyItemsd,
    SpatialPadd,
    EnsureChannelFirstd,
    Spacingd,
    OneOf,
    ScaleIntensityRanged,
    HistogramNormalized,
    RandSpatialCropSamplesd,
    RandSpatialCropd,
    CenterSpatialCropd,
    RandCoarseDropoutd,
    RandCoarseShuffled,
    Resized,
)

#%%
# basedir="."
basedir="/home/skowshik/ADRD_repo/pipeline_v1_main/adrd_tool"
cnf_file = f'{basedir}/dev/data/toml_files/default_conf.toml'
ckpt_path = f'/data_1/skowshik/ckpts_backbone_swinunet/current_best_model.pt'
img_mode=1
mri_type='ALL'

#%%

def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

flip_and_jitter = monai.transforms.Compose([
        monai.transforms.RandAxisFlipd(keys=["image"], prob=0.5),
        transforms.RandomApply(
            [
                monai.transforms.RandAdjustContrastd(keys=["image"], gamma=(-0.3,0.3)), # Random Gamma => randomly change contrast by raising the values to the power log_gamma 
                monai.transforms.RandBiasFieldd(keys=["image"]), # Random Bias Field artifact
                monai.transforms.RandGaussianNoised(keys=["image"]),

            ],
            p=0.4
        ),
    ])

# Custom transformation to filter problematic images
class FilterImages:
    def __init__(self, dat_type):
        # self.problematic_indices = []
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                CropForegroundd(keys=["image"], source_key="image"),
                monai.transforms.RandScaleCropd(keys=["image"], roi_scale=0.7, max_roi_scale=1, random_size=True, random_center=True),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=128),
                flip_and_jitter,
                monai.transforms.RandGaussianSmoothd(keys=["image"], prob=0.5),
                minmax_normalized,
            ]            
        )
        
        self.vld_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                CropForegroundd(keys=["image"], source_key="image"),
                # CenterSpatialCropd(keys=["image"], roi_size=(args.img_size,)*3),
                Resized(keys=["image"], spatial_size=(128*2,)*3),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=128),
                minmax_normalized,
            ]
        )
        
        if dat_type == 'trn':
            self.transforms = self.train_transforms
        else:
            self.transforms = self.vld_transforms

    def __call__(self, data):
        image_data = data["image"]
        try:
            return self.transforms(data)
        except Exception as e:
            print(f"Error processing image: {image_data}{e}")
            return None

# if using MRIs
# tst_filter_transform = FilterImages(dat_type='tst')

# for embeddings
tst_filter_transform = None

#%%
# Load the model
device = 'cuda:0'
img_dict = {'img_net': 'SwinUNETREMB', 'img_size': 128, 'patch_size': 16, 'imgnet_ckpt': ckpt_path, 'imgnet_layers': 4, 'train_imgnet': False}
mdl = ADRDModel.from_ckpt(ckpt_path, device=device, img_dict=img_dict)
print("loaded")

#%%
# dat_tst = CSVDataset(dat_file=dat_file, cnf_file=cnf_file, mode=0, img_mode=img_mode, mri_type=mri_type, other_3d_mris=other_3d_mris)

# TODO: modify cnf_file to have imaging modalities
# TODO: add data dict
# TODO: set values in skip_embedding to 'True' to skip the embedding modules, 'False' to use the embedding modules
import toml
cnf = toml.load(cnf_file)
data_dict = {}
skip_embedding = {}

for i in range(150):
    cnf['feature'][f'img_MRI_{i+1}'] = {'type': 'imaging', 'shape': "################ TO_FILL_MANUALLY ################"}

for k, info in cnf['feature'].items():
    skip_embedding[k] = False

# scores, scores_proba, y_pred = mdl.predict({'his_SEX':1}, _batch_size=1, img_transform=tst_filter_transform, skip_embedding=skip_embedding)
mdl.predict_logits([{'his_SEX':1}], _batch_size=1, skip_embedding={'img_MRI_1': True})


# %%
