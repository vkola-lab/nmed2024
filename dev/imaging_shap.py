
#%%

from data.imaging_data import CSVDataset
from adrd.model import ImagingModel

dat_file = '/home/skowshik/publication_ADRD_repo/adrd_tool/data/train_vld_test_split_updated/nacc_test_with_np_cli.csv'
cnf_file = 'data/toml_files/default_conf_new.toml'
ckpt_path = '/home/skowshik/publication_ADRD_repo/adrd_tool/dev/ckpt/ckpt_densenet_all_stripped_mni.pt'
# ckpt_path = '/data_1/skowshik/densenet_ckpt/bn_size_4/ckpt_densenet_t2_3way_config2.pt'
# ckpt_path = '/data_1/skowshik/densenet_ckpt/bn_size_4/ckpt_densenet_flair_3way_config2.pt'
# ckpt_path = '/data_1/skowshik/densenet_ckpt/bn_size_2/ckpt_densenet_other_3way_config2.pt'
emb_type = 'ALL' # 'T2', 'FLAIR', 'OTHER'
img_mode=0
mri_type="SEQ"
import shap

device = 'cuda:2'


import torch
import random
import json
from tqdm import tqdm
import numpy as np
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
from monai.utils.type_conversion import convert_to_tensor

# %%
backend='DenseNet'
mdl = ImagingModel.from_ckpt(ckpt_path, device=device, img_backend=backend, load_from_ckpt=True)
print("loaded")
mdl.net_.eval()

# %%
nacc_mri_info = "nacc_mri_3d.json"
other_mri_info = "other_3d_mris.json"
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
# %%
img_size = (182, 218, 182)
def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

# Custom transformation to filter problematic images
class FilterImages:
    def __init__(self, dat_type):
        
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
        # data = self.transforms(data)
        # print(data['image'].shape)
        try:
            data = self.transforms(data)
            # print(data.shape)
            if data['image'].shape[0] > 1:
                return None
            return data
        except Exception as e:
            # print(f"Error processing image: {image_data}{e}")
            return None
        
tst_filter_transform = FilterImages(dat_type='tst')
# %%
# initialize datasets
label_names = ['NC', 'MCI', 'DE']

dat = CSVDataset(dat_file=dat_file, label_names=label_names, mri_emb_dict=mri_emb_dict)
print("Loading Testing dataset ... ")
tst_list, df = dat.get_features_labels(mode=3)
#%%

# t1
# smp_idx = 30 # NC
# smp_idx = 36 # MCI
smp_idx = 1 # DE
print(tst_list[smp_idx]['label'])

smp = [dat.features[smp_idx]]
smp = torch.stack([convert_to_tensor(tst_filter_transform({"image": _})["image"]).to(f'{device}') for _ in smp])
out = mdl.net_(smp, shap=False)
print(out)


# %%
smp = [dat.features[smp_idx]]
print(smp)
smp = torch.stack([convert_to_tensor(tst_filter_transform({"image": _})["image"]).to(f'{device}') for _ in smp])
# smp = torch.stack([torch.tensor(_[src_k], device=device) for _ in smp])
# smp = convert_to_tensor(tst_filter_transform({"image": dat.features[smp_idx]})["image"]).to(device)

saliency_map = [np.zeros(img_size) for _ in range(3)]
# for _ in tqdm(range(n_runs)):
bkg = random.choices(dat.features, k=20)
bkg = torch.stack([convert_to_tensor(tst_filter_transform({"image": x})["image"]).to(f'{device}') for x in bkg])
# print(bkg)

with torch.set_grad_enabled(True):
    exp = shap.DeepExplainer(mdl.net_, bkg)
    shap_values = exp.shap_values(smp)

for i in range(3):
    saliency_map[i] += shap_values[i].squeeze()

# for i in range(3):
#     saliency_map[i] = saliency_map[i] / n_runs


# %%
# shap_values[0].shape
np.array(shap_values).shape
# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import ipywidgets as wd
from IPython.display import display

sal = saliency_map[2]
sal = gaussian_filter(sal, 2)
mri = tst_filter_transform({"image": dat.features[smp_idx]})["image"][0]
mri = np.array(mri)
sal = np.array(sal)

mri_min = np.min(mri)
mri_max = np.max(mri)
# mri = np.array((mri - mri_min) / (mri_max - mri_min))

sal_min = np.min(sal)
sal_max = np.max(sal)
# sal = np.array((sal - sal_min) / (sal_max - sal_min))
# print(mri.shape)
mri_axis = (0,1,2) # axial
# mri_axis = (0,2,1) # coronal
# mri_axis = (2,1,0) # sagittal
mri = mri.transpose(*mri_axis)
sal = sal.transpose(*mri_axis)

# Display MRI with "gray" colormap
%matplotlib widget
fig, axes = plt.subplots()
im_mri = axes.imshow(mri[:, :, 0], cmap="gray", alpha=1, origin="lower") #, vmax=mri.max(), vmin=sal.min())
im_sal = axes.imshow(sal[:, :, 0], cmap="coolwarm", alpha=0.5, origin="lower") #, vmax=1, vmin=0)  # Use alpha to control transparency

slider = wd.IntSlider(
    value=0,
    min=0,
    max=sal.shape[-1] - 1,
)

play_button = wd.Play(
    value=0,
    min=0,
    max=sal.shape[-1] - 1,
    step=1,
    interval=100,
    description="Press play",
)

wd.jslink((play_button, "value"), (slider, "value"))

def slider_update(change):
    im_mri.set_data(mri[:, :, change.new])
    im_sal.set_data(sal[:, :, change.new])
    fig.canvas.draw_idle()
    plt.suptitle(f'Time: {slider.value}')

slider.observe(slider_update, "value")

out = wd.Output()
app = wd.VBox([wd.HBox([play_button, slider]), out])
display(app)


# %%

# %%
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter
# slice_idx = 218 // 2

# # Show the selected slice 
# sal = saliency_map[4]
# sal = gaussian_filter(sal, 2)
# print(sal.shape)
# mri = tst_filter_transform({"image": dat.features[smp_idx]})["image"][0]
# # Overlay sal on mri
# overlay = np.zeros_like(mri)
# alpha = 0.999  # Adjust the transparency as needed
# overlay = (1 - alpha) * mri + alpha * sal
# # plt.imshow(img, cmap='coolwarm', vmin=-.001, vmax=.001)
# # plt.show()
# #%%
# sal.max()
# # %%
# import ipywidgets as wd
# npdata = overlay
# # cmap="gray"
# cmap="coolwarm"
# %matplotlib widget
# fig, axes = plt.subplots()
# axes.imshow(npdata[:,:,0], cmap=cmap, 
#         vmax=npdata.max(),
#         vmin=npdata.min(),)
# slider = wd.IntSlider(
#     value=0,
#     min=0,
#     max=npdata.shape[-1]-1,
# )

# play_button = wd.Play(
#     value=0,
#     min=0,
#     max=npdata.shape[-1]-1,
#     step=1,
#     interval=100,
#     description="Press play",
# )

# wd.jslink((play_button, "value"), (slider, "value"))

# def slider_update(change):
#     axes.imshow(
#         npdata[:,:,change.new],
#         cmap=cmap,
#         origin="lower",
#         vmax=npdata.max(),
#         vmin=npdata.min(),
#     )
#     fig.canvas.draw_idle()
#     plt.suptitle(f'Time: {slider.value}')

# slider.observe(slider_update, "value")

# out = wd.Output()
# app = wd.VBox([wd.HBox([play_button, slider]), out])
# display(app)
