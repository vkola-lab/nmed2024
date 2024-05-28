import pandas as pd
import numpy as np
from tqdm import tqdm
from adrd.nn import ImageModel
from icecream import ic
import torch
from .imgdataload import ImageDataset
from torch.utils.data import DataLoader
from monai.networks.nets.swin_unetr import SwinUNETR
import os
import random
import functools
from monai.utils.type_conversion import convert_to_tensor

ckpts_dict = {
    # 'unet3d': '~/adrd_tool/img_pretrained_ckpt/model_best.pkl',
    'SwinUNETR': '~/adrd_tool/dev/ssl_mri/pretrained_models/model_swinvit.pt'
}

device = 'cuda'

def get_mri_dataloader(feature, df, transforms=None, stripped=False):
    test_envs=[1,2,3]

    dataset = ImageDataset(feature=feature, task='mri_dg', root_dir='', domain_name="NACC_ADNI_NC_MCI_AD", domain_label=0, transform=transforms, indices=None, test_envs=test_envs, df=df, num_classes=3)
    
    def monai_collate_singles(samples_list, dataset, dtype=torch.half, return_dict=False, labels=None, multilabel=False):
        orig_len = len(samples_list)
        for s in samples_list:
            fname, img = s
            if s is None or img is None or img.shape[0] != 1 or torch.isnan(s[1]).any():
                samples_list.remove(s)

        # samples_list = [s for s in samples_list if not ]
        diff = orig_len - len(samples_list)
        ic(diff)
        if diff > 0:
            ic('recursive call')  
            return monai_collate_singles(samples_list + [dataset[random.randint(0, len(dataset)-1)] for _ in range(diff)], dataset, return_dict=return_dict, labels=labels, multilabel=multilabel)

        if return_dict:
            collated_dict = {"image": torch.stack([convert_to_tensor(s["image"]) for s in samples_list])}
            if labels:
                if multilabel:
                    collated_dict["label"] = {k: torch.Tensor([s["label"][k] for s in samples_list]) for k in labels}
                else:
                    collated_dict["label"] = torch.Tensor([s["label"] for s in samples_list])
            return collated_dict
        
        else:
            if isinstance(samples_list[0], tuple):
                # return fnames, imgs
                fnames_list = [s[0] for s in samples_list]
                imgs_list = [convert_to_tensor(s[1]) for s in samples_list]
                return fnames_list, torch.stack(imgs_list)
            return torch.stack([convert_to_tensor(s) for s in samples_list])

    collate_fn = functools.partial(monai_collate_singles, dataset=dataset)
    dataloader = DataLoader(dataset=dataset, batch_size=1, num_workers=1, drop_last=False, shuffle=False, collate_fn=collate_fn)

    return dataloader

def load_model(arch):
    ckpt_path = ckpts_dict[arch]
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # ic(state_dict.keys())
    if arch == 'SwinUNETR':
        model = SwinUNETR(
            in_channels=1,
            out_channels=1,
            img_size=128,
            feature_size=48,
            use_checkpoint=True,
        )
        ckpt["state_dict"] = {k.replace("swinViT.", "module."): v for k, v in ckpt["state_dict"].items()}
        ic(ckpt["state_dict"].keys())
        model.load_from(ckpt)
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model, *args, **kwargs) -> None:
                super().__init__(*args, **kwargs)
                self.swinunetr = model

            def forward(self, x_in):
                hidden_states_out = self.swinunetr.swinViT(x_in, self.swinunetr.normalize)
                ic(h.size() for h in hidden_states_out)
                enc0 = self.swinunetr.encoder1(x_in)
                enc1 = self.swinunetr.encoder2(hidden_states_out[0])
                enc2 = self.swinunetr.encoder3(hidden_states_out[1])
                enc3 = self.swinunetr.encoder4(hidden_states_out[2])
                dec4 = self.swinunetr.encoder10(hidden_states_out[4])
                dec3 = self.swinunetr.decoder5(dec4, hidden_states_out[3])
                dec2 = self.swinunetr.decoder4(dec3, enc3)
                dec1 = self.swinunetr.decoder3(dec2, enc2)
                dec0 = self.swinunetr.decoder2(dec1, enc1)
                out = self.swinunetr.decoder1(dec0, enc0)

                print(enc0.size(), enc1.size(), enc2.size(), enc3.size())
                print(dec4.size(), dec3.size(), dec2.size(), dec1.size(), dec0.size(), out.size())
                return dec4

        img_model = ModelWrapper(model)
    else:    
        state_dict = ckpt['model_dict']
        img_model = ImageModel(num_classes=3)
        img_model.load_checkpoint(state_dict)
    return img_model

def mri_emb(img_model, data):
    # print(data)
    fnames = data[0]
    # ic(len(fnames), fnames[0])
    x = data[1].float()
    y = data[2].long()
    # print(x.size(), y.size())
    x = x.to(device)
    y = y.to(device)
    # feats, output = img_model.predict(x, stage='get_features', attention=False)
    feats = img_model.extract_features(x, attention=True)
    # print(feats[1].shape)
    # break
    return np.array(feats[0].cpu())

def save_emb(feature, df):
    dataloader = get_mri_dataloader(feature, df)
    img_model = load_model()
    embeddings = []
    img_model.to(device)
    img_model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            try:
                filename = data[0][0].split('/')[-1]
                if os.path.exists('/home/skowshik/MRI_emb/' + filename):
                    continue
                emb = mri_emb(img_model, data)
                # print(emb.shape)
                np.save('/home/skowshik/MRI_emb/' + filename, emb)
            except:
                continue

def get_emb(feature, df, savedir, arch='unet3d', transforms=None, stripped=False):
    print('------------get_emb()------------')
    dataloader = get_mri_dataloader(feature, df, transforms=transforms, stripped=stripped)
    img_model = load_model(arch)
    # device = 'cuda'
    img_model.to(device)
    img_model.eval()
    embeddings = {}
    os.makedirs(savedir, exist_ok=True)
    with torch.no_grad():
        for fnames, data in tqdm(dataloader):
            try:
                if torch.isnan(data).any() or data.size(1) != 1:
                    continue
                data = data.float().to(device)
                # print(data.size())
                if arch == 'SwinUNETR':
                    emb = img_model(data)
                else:
                    emb = mri_emb(img_model, data)
                
                for idx, fname in enumerate(fnames):
                    print(fname)
                    if ('localizer' in fname.lower()) | ('localiser' in fname.lower()) |  ('LOC' in fname) | ('calibration' in fname.lower()) | ('field_mapping' in fname.lower()) | (fname.lower()[:-4].endswith('_ph_stripped')):
                        continue
                    # if 'DWI' in fname:
                    #     continue
                    if 'NACC' in fname:
                            filename = fname.split('/')[6] + '@' + '@'.join(fname.split('/')[-2:]).replace('.nii', '.npy')
                    elif 'FHS' in fname:
                        filename = 'FHS_' + '_'.join(fname.split('/')[-2:]).replace('.nii.gz', '.npy')
                    elif 'BMC' in fname:
                        filename = '_'.join(fname.split('/')[-3:]).replace('.nii', '.npy')
                    else:
                        filename = fname.split('/')[-1].replace('.nii', '.npy')
                    if os.path.exists(savedir + filename):
                        continue       
                    embeddings[filename] = emb[idx,:,:,:,:].cpu().detach().numpy()
                    np.save(savedir + filename, embeddings[filename])
            except:
                continue
    print("Embeddings saved to ", savedir)
    print("Done.")
    exit()
    return embeddings


