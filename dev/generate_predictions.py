# %%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import os
import datetime
import monai
import adrd.utils.misc

from tqdm import tqdm
from matplotlib.pyplot import figure
from torchvision import transforms
from icecream import ic
ic.disable()
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

from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel
from adrd.utils.misc import get_and_print_metrics_multitask
from adrd.utils.misc import get_metrics, print_metrics

#%%
# define paths and variables
basedir=".."

fname = 'name_of_file' # the name of file you want to save the model predictions to
save_path = f'./model_predictions/' # the path where the model predictions will be saved
dat_file = 'path/to/test/data' # the test data path
cnf_file = f'{basedir}/dev/data/toml_files/default_conf_new.toml' # the path configuration file
ckpt_path = 'path/to/saved/checkpoint' # the path to the model checkpoint

dat_file = pd.read_csv(dat_file)
print(dat_file)

# uncommment this to run without image embeddings
# img_net="NonImg"
# img_mode=-1

img_net="SwinUNETREMB"
img_mode=1

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
# load saved Transformer
device = 'cuda:0'
mdl = ADRDModel.from_ckpt(ckpt_path, device=device) 
print("loaded")

# %%
from matplotlib import rc, rcParams
rc('axes', linewidth=1)
rc('font', size=18)
plt.rcParams['font.family'] = 'Arial'

def read_csv(filename):
    return pd.read_csv(filename)

# Save model generated probabilities
def save_predictions(dat_tst, test_file, scores_proba, scores, save_path=None, filename=None, if_save=True):
    y_true = [{k:int(v) if v is not None else np.NaN for k,v in entry.items()} for entry in dat_tst.labels]
    mask = [{k:1 if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]

    y_true_ = {f'{k}_label': [smp[k] for smp in y_true] for k in y_true[0] if k in test_file.columns}
    scores_proba_ = {f'{k}_prob': [round(smp[k], 3) if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores_proba)] for k in scores_proba[0] if k in test_file.columns}
    scores_ = {f'{k}_logit': [round(smp[k], 3) if isinstance(y_true[i][k], int) else np.NaN for i, smp in enumerate(scores)] for k in scores[0] if k in test_file.columns}
    if 'RID' in list(test_file.columns):
        ids = test_file[['ID', 'RID']]
    else:
        ids = test_file['ID']

    y_true_df = pd.DataFrame(y_true_)
    scores_df = pd.DataFrame(scores_)
    scores_proba_df = pd.DataFrame(scores_proba_)
    if 'cdr_CDRGLOB' in test_file:
        cdr = test_file['cdr_CDRGLOB']
        cdr_df = pd.DataFrame(cdr)
    id_df = pd.DataFrame(ids)
    if 'fhs' in fname:
        fhsid = ids = test_file[['id', 'idtype', 'framid']]
        fhsid_df = pd.DataFrame(fhsid)
        if 'cdr_CDRGLOB' in test_file:
            df = pd.concat([fhsid_df, id_df, y_true_df, scores_proba_df, cdr_df], axis=1)
        else:
            df = pd.concat([fhsid_df, id_df, y_true_df, scores_proba_df], axis=1)
    else:
        if 'cdr_CDRGLOB' in test_file:
            df = pd.concat([id_df, y_true_df, scores_proba_df, cdr_df], axis=1)
        else:
            df = pd.concat([id_df, y_true_df, scores_proba_df], axis=1)
    print(len(y_true_df), len(scores_df), len(scores_proba_df), len(cdr_df), len(id_df))
    if if_save:
        df.to_csv(save_path + filename, index=False)
        
    return df

def generate_performance_report(dat_tst, y_pred, scores_proba):
    y_true = [{k:int(v) if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]
    mask = [{k:1 if v is not None else 0 for k,v in entry.items()} for entry in dat_tst.labels]

    y_true_dict = {k: [smp[k] for smp in y_true] for k in y_true[0]}
    y_pred_dict = {k: [smp[k] for smp in y_pred] for k in y_pred[0]}
    scores_proba_dict = {k: [smp[k] for smp in scores_proba] for k in scores_proba[0]}
    mask_dict = {k: [smp[k] for smp in mask] for k in mask[0]}

    met = {}
    for k in dat_tst.label_modalities:
        print('Performance metrics of {}'.format(k))
        metrics = get_metrics(np.array(y_true_dict[k]), np.array(y_pred_dict[k]), np.array(scores_proba_dict[k]), np.array(mask_dict[k]))
        print_metrics(metrics)

        met[k] = metrics
        met[k].pop('Confusion Matrix')

    return met

def generate_predictions_for_data_file(dat_file, labels, tst_filter_transform=None):
    # initialize datasets
    seed = 0
    print('Done.\nLoading testing dataset ...')
    dat_tst = CSVDataset(dat_file=dat_file, cnf_file=cnf_file, mode=0, img_mode=img_mode, stripped='_stripped_MNI')

    print('Done.')
    
    # generate model predictions
    print('Generating model predictions')
    scores, scores_proba, y_pred = mdl.predict(x=dat_tst.features, _batch_size=64, img_transform=tst_filter_transform)
    print('Done.')
    
    # save model predictions
    print('Saving model predictions')
    df = save_predictions(dat_tst, dat_file, scores_proba, scores, save_path, f'{fname}_prob.csv', if_save=True)
    print('Done.')
    
    # save performance report
    print('Generating performance reports')
    met = generate_performance_report(dat_tst, y_pred, scores_proba)
    print('Done.')
    return df, met
    
def generate_predictions_for_case(case_dict):
    return  mdl.predict(x=[test_case], _batch_size=1, img_transform=None)

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


#%%
if __name__ == '__main__':
    if img_mode in [0,2]:
        tst_filter_transform = FilterImages(dat_type='tst')
    else:
        tst_filter_transform = None
        
    # 1. Generate predictions for a test case file
    df_pred, met = generate_predictions_for_data_file(dat_file, labels, tst_filter_transform)
    #%%
    met_df = pd.DataFrame(met)
    met_df.to_csv(save_path + f'{fname}_performance_report.pdf', index=False)
    print(met_df.round(2))

    #%%
    # 2. Generate prediction for a single case
    # replace this dictionary with a dictionary of input features
    # test_case = {'his_NACCREAS': 0.0, 'his_NACCREFR': 2.0, 'his_SEX': 0, 'his_HISPANIC': 1, 'his_HISPOR': 1.0, 'his_RACE': 0, 'his_RACESEC': 3.0, 'his_PRIMLANG': 0.0, 'his_MARISTAT': 2.0, 'his_LIVSIT': 0.0, 'his_INDEPEND': 0.0, 'his_RESIDENC': 0.0, 'his_HANDED': 1.0, 'his_NACCNIHR': 5, 'his_NACCFAM': 1.0, 'his_NACCMOM': 0.0, 'his_NACCDAD': 1.0, 'his_NACCFADM': 0.0, 'his_NACCAM': 0.0, 'his_NACCFFTD': 0.0, 'his_NACCFM': 0.0, 'his_NACCOM': 0.0, 'his_TOBAC30': 0.0, 'his_TOBAC100': 1.0, 'his_CVHATT': 0.0, 'his_CVAFIB': 0.0, 'his_CVANGIO': 0.0, 'his_CVBYPASS': 0.0, 'his_CVPACE': 0.0} # example case
    
    # scores, scores_proba, y_pred = generate_predictions_for_case(test_case)
    # print(scores_proba)
