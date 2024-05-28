import torch
import torch.nn as nn
from typing import Any, Type
Tensor = Type[torch.Tensor]

from .resnet3d import r3d_18


class CNNResNet3D(nn.Module):

    def __init__(self,
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]]
    ) -> None:
        """ ... """
        super().__init__()

        # resnet
        # embedding modules for source
        self.modules_emb_src = nn.ModuleDict()
        for k, info in src_modalities.items():
            if info['type'] == 'imaging' and len(info['img_shape']) == 4:
                self.modules_emb_src[k] = nn.Sequential(
                    r3d_18(),
                    nn.Dropout(0.5)
                )
            else:
                # unrecognized
                raise ValueError('{} is an unrecognized data modality'.format(k))

        # classifiers (binary only)
        self.modules_cls = nn.ModuleDict()
        for k, info in tgt_modalities.items():
            if info['type'] == 'categorical' and info['num_categories'] == 2:
                # categorical
                self.modules_cls[k] = nn.Linear(256, 1)
            else:
                # unrecognized
                raise ValueError
            
    def forward(self,
        x: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """ ... """
        out_emb = self.forward_emb(x)
        out_emb = out_emb[list(out_emb.keys())[0]]
        out_cls = self.forward_cls(out_emb)
        return out_cls

    def forward_emb(self,
        x: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """ ... """
        out_emb = dict()
        for k in self.modules_emb_src.keys():
            out_emb[k] = self.modules_emb_src[k](x[k])
        return out_emb
    
    def forward_cls(self,
        out_emb: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        """ ... """
        out_cls = dict()
        for k in self.modules_cls.keys():
            out_cls[k] = self.modules_cls[k](out_emb).squeeze(1)
        return out_cls
    

# for testing purpose only
if __name__ == '__main__':
    src_modalities = {
        'img_MRI_T1': {'type': 'imaging', 'img_shape': [1, 182, 218, 182]}
    }
    tgt_modalities = {
        'AD': {'type': 'categorical', 'num_categories': 2},
        'PD': {'type': 'categorical', 'num_categories': 2}
    }
    net = CNNResNet3D(src_modalities, tgt_modalities)
    net.eval()
    x = {'img_MRI_T1': torch.zeros(2, 1, 182, 218, 182)}
    print(net(x))