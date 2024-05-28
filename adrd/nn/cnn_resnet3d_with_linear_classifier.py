import torch
import torch.nn as nn
from typing import Any, Type
Tensor = Type[torch.Tensor]

from .resnet3d import r3d_18

class CNNResNet3DWithLinearClassifier(nn.Module):

    def __init__(self,
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]]
    ) -> None:
        """ ... """
        super().__init__()
        self.core = _CNNResNet3DWithLinearClassifier(len(tgt_modalities))
        self.src_modalities = src_modalities
        self.tgt_modalities = tgt_modalities

    def forward(self,
        x: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """ x is expected to be a singleton dictionary """
        src_k = list(x.keys())[0]
        x = x[src_k]
        out = self.core(x)
        out = {tgt_k: out[:, i] for i, tgt_k in enumerate(self.tgt_modalities)}
        return out


class _CNNResNet3DWithLinearClassifier(nn.Module):

    def __init__(self,
        len_tgt_modalities: int,
    ) -> None:
        """ ... """
        super().__init__()
        self.cnn = r3d_18()
        self.cls = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256, len_tgt_modalities),
        )

    def forward(self, x: Tensor) -> Tensor:
        """ ... """
        out_emb = self.forward_emb(x)
        out_cls = self.forward_cls(out_emb)
        return out_cls
    
    def forward_emb(self, x: Tensor) -> Tensor:
        """ ... """
        return self.cnn(x)
    
    def forward_cls(self, out_emb: Tensor) -> Tensor:
        """ ... """
        return self.cls(out_emb)