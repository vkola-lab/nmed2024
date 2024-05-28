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

import torch
import torch.nn as nn

from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.utils import ensure_tuple_rep

class SSLHead(nn.Module):
    def __init__(self, args, upsample="vae", dim=48):
        super(SSLHead, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.swinunetr = SwinUNETR(
            in_channels=args.in_channels,
            out_channels=1,
            img_size=(args.roi_x,args.roi_y,args.roi_z),
            feature_size=args.feature_size,
            use_checkpoint=True,
        )
        pretrained_pth = "~/dev/ssl_mri/pretrained_models/model_swinvit.pt"
        model_dict = torch.load(pretrained_pth, map_location="cpu")
        model_dict["state_dict"] = {k.replace("swinViT.", "module."): v for k, v in model_dict["state_dict"].items()}
        ic(model_dict["state_dict"].keys())
        self.swinunetr.load_from(model_dict)

        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)

    def forward(self, x_in, output_only=False):
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
        # logits = self.out(out)
        # return logits
        ic(enc0.size(), enc1.size(), enc2.size(), enc3.size())
        ic(dec4.size(), dec3.size(), dec2.size(), dec1.size(), dec0.size())
        _, c, h, w, d = out.shape
        x4_reshape = out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        x_rot = self.rotation_pre(x4_reshape[:, 0])
        x_rot = self.rotation_head(x_rot)
        x_contrastive = self.contrastive_pre(x4_reshape[:, 1])
        x_contrastive = self.contrastive_head(x_contrastive)
        x_rec = self.swinunetr.out(out)
        if output_only:
            return x_rec
        return x_rot, x_contrastive, x_rec