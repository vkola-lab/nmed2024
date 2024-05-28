#%%
import pandas as pd
from adrd.shap_adrd import MCExplainer
from adrd.utils import TransformerValidationDataset, TransformerTestingDataset
from data.dataset_csv import CSVDataset
from adrd.model import ADRDModel
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from torchvision import transforms
import os

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
from icecream import ic, install
install()
#%%
prefix = "/projectnb/ivc-ml/dlteif/debug_adrd/adrd_tool"
test_path = f"{prefix}/data/train_vld_test_split_updated/nacc_test_with_np_cli.csv"
cnf_file = f"{prefix}/dev/data/toml_files/default_conf.toml"
ckpt_path = f'{prefix}/dev/ckpt/new_embeddings_current_best_model_correction.pt'
imgnet_ckpt= '/projectnb/ivc-ml/dlteif/pretrained_models/model_swinvit.pt'
img_size=128

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

#%%

if __name__ == '__main__':

    dat_tst = CSVDataset(dat_file=test_path, cnf_file=cnf_file, mode=2, img_mode=1, mri_type='ALL', other_3d_mris=None, arch='SwinUNETREMB', transforms=tst_transforms, emb_path="/projectnb/ivc-ml/dlteif/SwinUNETR_MRI_stripped_emb/", stripped=True)

    label_fractions = dat_tst.label_fractions

    df = pd.read_csv(test_path)
    label_distribution = {}
    for label in ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']:
        label_distribution[label] = dict(df[label].value_counts())

    print(label_fractions)
    print(label_distribution)
#%%
    # %%
    # initialize and save Transformer
    mdl = ADRDModel(
        src_modalities = dat_tst.feature_modalities,
        tgt_modalities = dat_tst.label_modalities,
        label_fractions = label_fractions,
        d_model = 256,
        nhead = 1,
        num_epochs = 256,
        batch_size = 1, # 64, 
        weight_decay = 0.01,
        criterion = 'AUC (ROC)',
        device = 'cuda',
        cuda_devices = [0],
        img_net = 'SwinUNETREMB',
        imgnet_layers = 4,
        img_size = img_size,
        fusion_stage= 'middle',
        imgnet_ckpt = imgnet_ckpt,
        patch_size = 16,
        ckpt_path = ckpt_path,
        train_imgnet = False,
        load_from_ckpt = True,
        save_intermediate_ckpts = True,
        data_parallel = False,
        verbose = 4,
        wandb_ = False,
        label_distribution = label_distribution,
        ranking_loss = True,
        # k = 5,
        _amp_enabled = False,
        _dataloader_num_workers = 1,
    )
    mdl._init_net()
    torch.set_num_threads(1)
    
    print(type(tst_transforms))
    dat_tst = TransformerValidationDataset(
            dat_tst.features, dat_tst.labels,
            mdl.src_modalities, mdl.tgt_modalities,
            img_transform=tst_transforms,
        )
    
    # dat_tst = TransformerTestingDataset(
    #         dat_tst.features,
    #         mdl.src_modalities,
    #         img_transform=tst_transforms,
    #     )
    
    ldr_tst = DataLoader(
            dataset = dat_tst,
            batch_size = 1,
            shuffle = False,
            drop_last = False,
            num_workers = mdl._dataloader_num_workers,
            collate_fn = TransformerValidationDataset.collate_fn,
            # pin_memory = True
        )

    print('Test Loader initialized.')
    print(len(ldr_tst), len(ldr_tst.dataset))

    # if os.path.exists('./datamean.txt'):
    #     explainer = MCExplainer.from_ckpt(mdl, './datamean.txt')
    # else:
    explainer = MCExplainer(mdl)

    # print(type(explainer.data_mean))
    # print(mdl.tgt_modalities.keys(), mdl.src_modalities.keys())
    phi = explainer.shap_values(ldr_tst)
    