from abc import ABC
from abc import abstractmethod
from typing import Any, Type
from functools import wraps
from torch.utils.data import DataLoader
from torch import set_grad_enabled
import torch
Tensor = Type[torch.Tensor]

from ..utils.misc import convert_args_kwargs_to_kwargs
from ..utils import TransformerTestingDataset
from ..model import Transformer

class BaseExplainer:
    """ ... """
    def __init__(self, model: Transformer) -> None:
        """ ... """
        self.model = model

    def shap_values(self, 
        x,
        is_embedding: dict[str, bool] | None = None,
    ):
        """ ... """
        # result placeholder
        phi = [
            {
                tgt_k: {
                    src_k: 0.0 for src_k in self.model.src_modalities
                } for tgt_k in self.model.tgt_modalities
            }
        ]

        # set nn to eval mode
        set_grad_enabled(False)
        self.model.net_.eval()

        # initialize dataset and dataloader object
        dat = TransformerTestingDataset(x, self.model.src_modalities, is_embedding)
        ldr = DataLoader(
            dataset = dat,
            batch_size = 1,
            shuffle = False,
            drop_last = False,
            num_workers = 0,
            collate_fn = TransformerTestingDataset.collate_fn,
        )

        # loop through instances and compute shap values
        for idx, (smp, mask) in enumerate(ldr):
            mask_flat = torch.concatenate(list(mask.values()))
            if torch.logical_not(mask_flat).sum().item() == 0:
                pass
            elif torch.logical_not(mask_flat).sum().item() == 1:
                pass
            else:
                self._shap_values_core(smp, mask, phi[idx], is_embedding)

        return phi

    @abstractmethod
    def _shap_values_core(self,
        smp: dict[str, Tensor], 
        mask: dict[str, Tensor],
        phi_: dict[str, dict[str, float]],
    ):
        """ To implement different algorithms. """
        pass