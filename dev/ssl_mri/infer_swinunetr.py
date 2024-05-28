# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial

import nibabel as nib
import numpy as np
import torch
from utils.data_utils import minmax_normalized, monai_collate_singles, monai_collate
from data.mri_dataset import get_fpaths, MonaiDataset
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
from monai.data import DataLoader
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    RandSpatialCropSamplesd,
    Spacingd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)
from models.ssl_head import SSLHead
from main_swinunetr import get_parser
import icecream
from icecream import install, ic
install()
ic.configureOutput(includeContext=True)
ic.enable()

parser = get_parser()
parser.add_argument("--data_dir", default="/dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument("--fold", default=1, type=int, help="data fold")
parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
parser.add_argument("--infer_overlap", default=0.6, type=float, help="sliding window inference overlap")
parser.add_argument("--out_channels", default=3, type=int, help="number of output channels")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument(
    "--pretrained_dir",
    default="/projectnb/ivc-ml/dlteif/pretrained_models/",
    type=str,
    help="pretrained checkpoint directory",
)


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./outputs/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    test_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            # minmax_normalized,
            # ScaleIntensityRanged(
            #     keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            # ),
            SpatialPadd(keys="image", spatial_size=[args.roi_x, args.roi_y, args.roi_z]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]),
            RandSpatialCropSamplesd(
                 keys=["image"],
                 roi_size=[args.roi_x, args.roi_y, args.roi_z],
                 num_samples=2,
                 random_center=False,
                 random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )

    data_list = get_fpaths(args.data_dir, stripped=True)
    test_ds = MonaiDataset(data=data_list, transform=test_transforms)
    ic(len(test_ds))
    collate_fn = partial(monai_collate, dataset=test_ds)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0, shuffle=False, drop_last=True, collate_fn=collate_fn)
    ic(len(test_loader))
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    # model = SSLHead(args)
    model = SwinUNETR(
            in_channels=1,
            out_channels=1,
            img_size=(args.roi_x,args.roi_y,args.roi_z),
            feature_size=48,
            use_checkpoint=True,
        )
    ic(sum(p.numel() for p in model.parameters()))
    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=4,
        predictor=model,
        overlap=args.infer_overlap,
        )
        
    model_dict = torch.load(pretrained_pth, map_location="cpu")
    #ic(model_dict["state_dict"].keys())
    #model_dict["state_dict"] = {k.replace("module.swinunetr.", "module.").replace("swinunetr.", "module.").replace("module.swinViT.", "module."): v for k, v in model_dict["state_dict"].items() if "swinunetr" in k}
    #model_dict["state_dict"] = {k.replace("mlp.linear", "mlp.fc"): v for k, v in model_dict["state_dict"].items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    # model_dict["state_dict"] = {k: v for k, v in model_dict["state_dict"].items() if "swinunetr" in k}
    #ic(model_dict["state_dict"].keys())
    model.load_from(model_dict)
    # model.load_state_dict(model_dict["state_dict"])
    model.eval()
    model.to(device)

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.model = model

        def forward(self, x):
            output = self.model(x, output_only=True)
            print(output.size(), output.min(), output.max())
            return output

    # predictor = ModelWrapper(model)

    model_inferer_test = partial(
        sliding_window_inference,
        roi_size=[args.roi_x, args.roi_y, args.roi_z],
        sw_batch_size=1,
        predictor=model,
        overlap=args.infer_overlap,
    )

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            image = batch.cuda()
            print(image.size())
            # affine = batch["image_meta_dict"]["original_affine"][0].numpy()
            # num = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1].split("_")[1]
            # print("Inference on case {}".format(img_name))
            #prob = model_inferer_test(image)
            _, prob = model(image)
            print(prob.size(), prob.min(), prob.max())
            #prob = minmax_normalized({"image": prob})["image"]
            print(prob.size(), prob.min(), prob.max())
            # seg = prob[0].detach().cpu().numpy()
            # seg = (seg > 0.5).astype(np.int8)
            # seg_out = np.zeros((seg.shape[1], seg.shape[2], seg.shape[3]))
            # seg_out[seg[1] == 1] = 2
            # seg_out[seg[0] == 1] = 1
            # seg_out[seg[2] == 1] = 4
            np.save(f'outputs/orig_{str(i)}.npy', image.detach().cpu().numpy())
            np.save(f'outputs/prob_{str(i)}.npy', prob.detach().cpu().numpy())
            #nib.save(nib.Nifti1Image(image.detach().cpu().numpy(), affine=np.eye(4)), f'outputs/orig_{str(i)}.nii')
            #nib.save(nib.Nifti1Image(prob.detach().cpu().numpy(), affine=np.eye(4)), f'outputs/recon_{str(i)}.nii')
            if i > 5:
                break
        print("Finished inference!")


if __name__ == "__main__":
    main()