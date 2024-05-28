__all__ = ['MCExplainer']

from . import BaseExplainer
from typing import Any, Type
from torch import set_grad_enabled
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import random
import torch
import numpy as np
from tqdm import tqdm
Tensor = Type[torch.Tensor]

NUM_PERMUTATIONS = 1024
BATCH_SIZE = NUM_PERMUTATIONS

class MCExplainer(BaseExplainer):

    def __init__(self,
        model: Any,
    ):
        """ ... """
        super().__init__(model)

    def _shap_values_core(self,
        smp: dict[str, Tensor],
        mask: dict[str, Tensor],
        phi_: dict[str, dict[str, float]],
        is_embedding: dict[str, bool] | None = None,
    ):
        """ ... """
        # get the list of available feature names
        avail = [k for k in mask if mask[k].item() == False]

        # repeat feature dict and mount to device
        smps = dict()
        for k, v in smp.items():
            if len(v.shape) == 1:
                smps[k] = smp[k].repeat(NUM_PERMUTATIONS)
            else:
                smps[k] = smp[k].repeat(NUM_PERMUTATIONS, 1)
        smps = {k: smps[k].to(self.model.device) for k in self.model.src_modalities}
        
        # loop through available features
        print('{} features to evaluate ...'.format(len(avail)))
        for src_k in tqdm(avail):
            # get features to uncover
            to_uncover = []
            for _ in range(NUM_PERMUTATIONS):
                perm = avail.copy()
                random.shuffle(perm)
                to_uncover.append(perm[:perm.index(src_k)])

            # construct masks without src_k
            masks_wo_src_k = {k: np.ones(NUM_PERMUTATIONS, dtype=np.bool_) for k in self.model.src_modalities}
            for i, lst in enumerate(to_uncover):
                for k in lst:
                    masks_wo_src_k[k][i] = False

            # construct masks with src_k
            masks_wi_src_k = masks_wo_src_k.copy()
            masks_wi_src_k[src_k] = np.zeros(NUM_PERMUTATIONS, dtype=np.bool_)

            # mount inputs to device
            masks_wi_src_k = {k: torch.tensor(masks_wi_src_k[k], device=self.model.device) for k in self.model.src_modalities}
            masks_wo_src_k = {k: torch.tensor(masks_wo_src_k[k], device=self.model.device) for k in self.model.src_modalities}

            # run model
            out_wi_src_k = self.model.net_(smps, masks_wi_src_k, is_embedding)
            out_wo_src_k = self.model.net_(smps, masks_wo_src_k, is_embedding)

            # to numpy
            out_wi_src_k = {k: out_wi_src_k[k].cpu().numpy() for k in self.model.tgt_modalities}
            out_wo_src_k = {k: out_wo_src_k[k].cpu().numpy() for k in self.model.tgt_modalities}

            # replace nan with zeros when all input features are excluded
            out_wo_src_k = {k: np.nan_to_num(out_wo_src_k[k]) for k in self.model.tgt_modalities}

            # calculate shap values
            mean = {k: (out_wi_src_k[k] - out_wo_src_k[k]).mean() for k in self.model.tgt_modalities}   
            for tgt_k in self.model.tgt_modalities:
                phi_[tgt_k][src_k] = mean[tgt_k]