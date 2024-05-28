#%%
import os
from glob import glob
import pandas as pd
from subprocess import call
from tqdm import tqdm
from data.dataset_csv import CSVDataset
import monai
from monai.transforms import (
	Compose,
	LoadImaged,
	Resized,
	Spacingd,
	EnsureChannelFirstd,
	CropForegroundd,
	ResizeWithPadOrCropd,
)

prefix = "/projectnb/ivc-ml/dlteif/debug_adrd/adrd_tool"
test_path = f'{prefix}/data/training_cohorts/train_vld_test_split_updated/merged_vld.csv'
cnf_file = f"{prefix}/dev/data/toml_files/default_conf.toml"
img_size=128
# df = pd.readcsv(test_path)
#%%
def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

tst_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                CropForegroundd(keys=["image"], source_key="image"),
                # CenterSpatialCropd(keys=["image"], roi_size=(args.img_size,)*3),
                Resized(keys=["image"], spatial_size=(img_size*2,)*3),
                monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=img_size),
                minmax_normalized,
            ]
        )

dat_tst = CSVDataset(dat_file=test_path, cnf_file=cnf_file, mode=2, img_mode=0, mri_type='ALL', other_3d_mris=None, arch='SwinUNETR', transforms=tst_transforms, stripped=False)
#%%
fpaths = []
for smp in dat_tst.features:
	for k,v in smp.items():
		if 'img_MRI' in k and v is not None and '_stripped' not in v and 'SEG' not in v:
			fpaths.append(v)
#%%

# with open("./fpaths.txt", "r") as f:
# 	fpaths = f.read().splitlines()
fpaths = [fpath for fpath in fpaths if not('_e2' in fpath or 'localizer' in fpath.lower() or 'calibration' in fpath.lower())]
# 	fpaths = [fpath for fpath in fpaths if 'DWI' in fpath][::-1]
for fpath in tqdm(fpaths[::-1]):
	print(fpath)
	print(os.path.exists(fpath.replace('.nii', '_stripped.nii')))
	if os.path.exists(fpath.replace('.nii', '_stripped.nii')):
		continue
	esc_fpath = fpath.replace("(", "\(").replace(")", "\)").replace(" ", "\ ")
	call(f"/projectnb/ivc-ml/dlteif/synthstrip-singularity -i \'{esc_fpath}\' -o \'{esc_fpath.replace('.nii', '_stripped.nii')}\' -m \'{esc_fpath.replace('.nii', '_stripped_mask.nii')}\'", shell=True)
	
	# if os.path.exists(fpath.replace('.nii', '_stripped_nocsf.nii')):
	# 	continue
	# call(f"/projectnb/ivc-ml/dlteif/synthstrip-singularity -i \'{esc_fpath}\' -o \'{esc_fpath.replace('.nii', '_stripped_nocsf.nii')}\' -m \'{esc_fpath.replace('.nii', '_stripped_nocsf_mask.nii')}\' --no-csf", shell=True)
# f.close()

# %%
