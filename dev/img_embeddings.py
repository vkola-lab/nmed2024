#%%
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import argparse
import json
import os
import adrd.utils.misc
import random
import monai
import nibabel as nib

from tqdm import tqdm
from torchvision import transforms
from adrd.utils.misc import get_and_print_metrics_multitask
from adrd.utils.misc import get_metrics, print_metrics
from sklearn.metrics import multilabel_confusion_matrix, classification_report, roc_curve, auc, confusion_matrix, \
     RocCurveDisplay, precision_score, recall_score, average_precision_score, PrecisionRecallDisplay, precision_recall_curve
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
from icecream import ic, install
install()
ic.configureOutput(includeContext=True)
ic.disable()
from matplotlib.pyplot import figure
from data.imaging_data import CSVDataset
from adrd.model import ImagingModel


#%%
# define variables
prefix = '/home/skowshik/ADRD_repo/pipeline_v1_main/adrd_tool/data/train_vld_test_split_updated'

dat_files = [f'{prefix}/merged_train.csv', f'{prefix}/merged_vld.csv', f'{prefix}/nacc_test_with_np_cli.csv', f'{prefix}/adni_revised_labels.csv']
# dat_files = [f'{prefix}/adni_revised_labels.csv']
ckpt_path = '/home/skowshik/publication_ADRD_repo/adrd_tool/dev/ckpt/ckpt_densenet_232_all_stripped_mni.pt'
emb_type = 'ALL'
save_path = '/data_1/skowshik/DenseNet_emb_new/'
img_backend='DenseNet'
img_size = (182,218,182)
batch_size = 16
label_names = ['NC', 'MCI', 'DE']

nacc_mri_info = "dev/nacc_mri_3d.json"
other_mri_info = "dev/other_3d_mris.json"

# nacc_mri_info = "nacc_mri_3d.json"
# other_mri_info = "other_3d_mris.json"

other_path = '/SeaExpCIFS/Raw_MRIs/ALL_nii'
sequence_list = ['T1', 'T2', 'FLAIR', 'SWI', 'DTI', 'DWI']
stripped = '_stripped_MNI'
cnt = 0
    
# load 3d mris list for nacc
with open(nacc_mri_info) as json_data:
    nacc_mri_json = json.load(json_data)
    
# load 3d mris list for other cohorts          
with open(other_mri_info) as json_data:
    other_mri_json = json.load(json_data)

other_3d_mris = []
for sequence in sequence_list:
    other_3d_mris += other_mri_json[sequence.lower()]
    
#%%

avail_cohorts = set()     
mri_emb_dict = {}
                    
for mag, seq in nacc_mri_json.items():
    for seq_name, mri_name in tqdm(seq.items()):
        if seq_name.upper() not in sequence_list:
            continue
        if emb_type != "ALL":
            if emb_type.lower() != seq_name.lower():
                continue
        for name, pairs in mri_name.items():
            for pair in pairs:
                mri = pair['mri']
                if 't1' in mri.lower() and 'MT1' in mri.lower():
                    continue
                if stripped and not f'{stripped}.nii' in mri:
                    if not os.path.exists(mri.replace('.nii', f'{stripped}.nii')):
                        continue
                    # print("here")
                    mri = mri.replace('.nii', f'{stripped}.nii')
                    
                if ('localizer' in mri.lower()) | ('localiser' in mri.lower()) |  ('LOC' in mri) | ('calibration' in mri.lower()) | ('field_mapping' in mri.lower()) | ('_ph' in mri.lower()) | ('seg' in mri.lower()) | ('aahscout' in mri.lower()) | ('aascout' in mri.lower()):
                    continue
                zip_name = name[:-2] + '.zip'
                if zip_name in mri_emb_dict:
                    mri_emb_dict[zip_name].add(mri)
                else:
                    mri_emb_dict[zip_name] = set()
                    mri_emb_dict[zip_name].add(mri)
                cnt += 1
                
mri_emb_dict = {k : list(v) for k, v in mri_emb_dict.items()}
# print(mri_emb_dict)

if len(mri_emb_dict) != 0:
    avail_cohorts.add('NACC')
    print("here")

# %%
# other cohorts
for cohort in os.listdir(other_path):
    if os.path.isfile(f'{other_path}/{cohort}'):
        continue
    
    for mri in tqdm(os.listdir(f'{other_path}/{cohort}')):
        if mri.endswith('json'):
            continue
                
        if stripped and not f'{stripped}.nii' in mri:
            if not os.path.exists(mri.replace('.nii', f'{stripped}.nii')):
                continue
            mri = mri.replace('.nii', f'{stripped}.nii')
        
        # remove localizers 
        if ('localizer' in mri.lower()) | ('localiser' in mri.lower()) |  ('LOC' in mri) | ('calibration' in mri.lower()) | ('field_mapping' in mri.lower()) | ('_ph' in mri.lower()) | ('seg' in mri.lower()) | ('aahscout' in mri.lower()) | ('aascout' in mri.lower()):
            continue
        
        # remove 2d mris
        if other_3d_mris is not None and len(other_3d_mris) != 0 and not mri.replace(f'{stripped}.nii', '.nii') in other_3d_mris:
            continue
        
        # select mris of sequence seq_type for SEQ img_mode
        if emb_type != "ALL":
            if mri.replace(f'{stripped}.nii', '.nii') not in other_mri_json[seq_type.lower()]:
                continue
        
            
        if (mri.lower().startswith('adni')) or (mri.lower().startswith('nifd')) or (mri.lower().startswith('4rtni')):
            name = '_'.join(mri.split('_')[:4])
        elif (mri.lower().startswith('aibl')) or (mri.lower().startswith('sub')) or (mri.lower().startswith('ppmi')):
            name =  '_'.join(mri.split('_')[:2])
        elif mri.lower().startswith('stanford') or 'stanford' in cohort.lower():
            if 't1' in mri.lower():
                name = mri.split('.')[0].split('_')[0] + '_' + mri.split('.')[0].split('_')[2]
            else:
                name = mri.split('.')[0]
        else:
            continue
        
        # if name.split('_')[0] == 'ADNI':
            # print('found')
        if mri.lower().startswith('sub'):
            avail_cohorts.add('OASIS')
        else:
            avail_cohorts.add(name.split('_')[0])
        
        if name in mri_emb_dict:
            mri_emb_dict[name.replace(f'{stripped}', '')].add(f'{other_path}/{cohort}/{mri}')
        else:
            mri_emb_dict[name.replace(f'{stripped}', '')] = set()
            mri_emb_dict[name.replace(f'{stripped}', '')].add(f'{other_path}/{cohort}/{mri}')
        cnt += 1
          
mri_emb_dict = {k : list(v) for k, v in mri_emb_dict.items()}

print("AVAILABLE MRI Cohorts: ", avail_cohorts)
if 'NACC' not in avail_cohorts:
    print('NACC MRIs not available')
print(f"Avail mris: {cnt}")
    
#%%
def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

# Custom transformation to filter problematic images
class FilterImages:
    def __init__(self, dat_type):
        # self.problematic_indices = []
        
        self.tst_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                # CropForegroundd(keys=["image"], source_key="image"),
                # Resized(keys=["image"], spatial_size=img_size),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
                minmax_normalized,
            ]
        )
        
        self.transforms = self.tst_transforms

    def __call__(self, data):
        try:
            image_data = data["image"]
            check = nib.load(image_data).get_fdata()
            if len(check.shape) > 3:
                return None
            return self.transforms(data)
        except Exception as e:
            print(f"Error processing image: {image_data}{e}")
            return None
        
        
tst_filter_transform = FilterImages(dat_type='tst')

# print(tst_filter_transform({'image': '/SeaExpCIFS/Raw_MRIs/ALL_nii/STANFORD_nii/3.nii'})['image'])

#%%
# load the model and the dataloader
device = 'cuda:2'
mdl = ImagingModel.from_ckpt(ckpt_path, device=device, img_backend=img_backend, load_from_ckpt=True, how='latest')
print("loaded")

torch.set_grad_enabled(False)
mdl.net_.eval()
# mdl.to(device)
#%%
seed = 0
label_names = ['NC', 'MCI', 'DE']
# initialize datasets
avail_cohorts = {}
for dat_file in dat_files:
    print(dat_file)
    dat = CSVDataset(dat_file=dat_file, label_names=label_names, mri_emb_dict=mri_emb_dict)
    print("Loading Testing dataset ... ")
    tst_list, df = dat.get_features_labels(mode=3)

    logits: list[dict[str, float]] = []
    img_embeddings = None
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i, fea_dict in tqdm(enumerate(tst_list)):
        image = fea_dict['image']
        f = image.split('/')
        # print(f)
        
        if 'NACC' in image:
            emb_path = save_path + '@'.join(f[4:]).replace('.nii', '.npy')
        else:
            emb_path = save_path + f[-1].replace('.nii', '.npy')
        if '@' in emb_path:
            cohort_name = 'NACC'
        else:
            # print(f)
            cohort_name = f[4].split('_')[0]
            
        if cohort_name in avail_cohorts:
            avail_cohorts[cohort_name] += 1
        else:
            avail_cohorts[cohort_name] = 1
        # break
        if os.path.exists(emb_path):
            print("Available")
            continue
        print('save_path:' + emb_path)
        print('img_path:' + image)
        
        
        try:
            mri = tst_filter_transform({'image': image})['image']
            print(mri.shape)
            outputs = torch.nn.Sequential(*list(mdl.net_.features.children()))(mri.unsqueeze(0).to(mdl.device))
            outputs = torch.flatten(outputs, 1)
            # outputs = outputs.squeeze(0)
            print(outputs.shape)
            try:
                np.save(emb_path, outputs)
            except:
                np.save(save_path + '@'.join([f[4], f[-1]]).replace('.nii', '.npy'), outputs)
            print("saved")
        except:
            continue
        # break
    # break
print(avail_cohorts)

 #%%
# dat = CSVDataset(dat_file=dat_files[0], label_names=label_names, mri_emb_dict=mri_emb_dict)
# print("Loading Testing dataset ... ")
# tst_list, df = dat.get_features_labels(mode=3)
# ldr_tst = mdl._init_test_dataloader(batch_size=batch_size, tst_list=tst_list, img_tst_trans=tst_filter_transform)

# #%%
# scores, scores_proba, y_pred = mdl.predict(ldr_tst)

# #%%

# def save_performance_report(met, filepath):
#     figure(figsize=(24, 20), dpi=300)
#     met_df = pd.DataFrame(met).transpose()
#     report_plot = sns.heatmap(met_df, annot=True)

#     # plt.show()
#     report_plot.figure.savefig(filepath, format='pdf', dpi=300, bbox_inches='tight')

# # list-of-dict to dict-of-list
# y_true = [{k:int(v) if v is not None else 0 for k,v in entry.items()} for entry in dat.labels]
# mask = [{k:1 if v is not None else 0 for k,v in entry.items()} for entry in dat.labels]
# y_true_dict = {k: [smp[k] for smp in y_true] for k in y_true[0]}
# y_pred_dict = {k: [smp[k] for smp in y_pred] for k in y_pred[0]}
# scores_proba_dict = {k: [smp[k] for smp in scores_proba] for k in scores_proba[0]}
# mask_dict = {k: [smp[k] for smp in mask] for k in mask[0]}

# met = {}
# for k in label_names:
#     # if k in ['NC', 'MCI', 'DE']:
#     #     continue
#     print('Performance metrics of {}'.format(k))
#     metrics = get_metrics(np.array(y_true_dict[k]), np.array(y_pred_dict[k]), np.array(scores_proba_dict[k]), np.array(mask_dict[k]))
#     print_metrics(metrics)

#     met[k] = metrics
#     met[k].pop('Confusion Matrix')

# save_performance_report(met, f'/home/skowshik/publication_ADRD_repo/adrd_tool/densenet_plots/performane_report_{emb_type}.pdf')


# %%
