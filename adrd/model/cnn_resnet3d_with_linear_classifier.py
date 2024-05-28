__all__ = ['CNNResNet3DWithLinearClassifier']

import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from scipy.special import expit
from copy import deepcopy
from contextlib import suppress
from typing import Any, Self, Type
from functools import wraps
Tensor = Type[torch.Tensor]
Module = Type[torch.nn.Module]

from ..utils.misc import ProgressBar
from ..utils.misc import get_metrics_multitask, print_metrics_multitask

from .. import nn
from ..utils import TransformerTrainingDataset
from ..utils import Transformer2ndOrderBalancedTrainingDataset
from ..utils import TransformerValidationDataset
from ..utils import TransformerTestingDataset
from ..utils.misc import ProgressBar
from ..utils.misc import get_metrics_multitask, print_metrics_multitask
from ..utils.misc import convert_args_kwargs_to_kwargs


def _manage_ctx_fit(func):
    ''' ... '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        # format arguments
        kwargs = convert_args_kwargs_to_kwargs(func, args, kwargs)

        if kwargs['self']._device_ids is None:
            return func(**kwargs)
        else:
            # change primary device
            default_device = kwargs['self'].device
            kwargs['self'].device = kwargs['self']._device_ids[0]
            rtn = func(**kwargs)

            # the actual module is wrapped
            kwargs['self'].net_ = kwargs['self'].net_.module
            kwargs['self'].to(default_device)
            return rtn
    return wrapper


class CNNResNet3DWithLinearClassifier(BaseEstimator):

    def __init__(self,
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]],
        num_epochs: int = 32,
        batch_size: int = 8,
        batch_size_multiplier: int = 1,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        beta: float = 0.9999,
        gamma: float = 2.0,
        scale: float = 1.0,
        criterion: str | None = None,
        device: str = 'cpu',
        verbose: int = 0,
        _device_ids: list | None = None,
        _dataloader_num_workers: int = 0,
        _amp_enabled: bool = False,
        _tmp_ckpt_filepath: str | None = None,
    ) -> None:  
        ''' ... '''
        # for multiprocessing
        self._rank = 0
        self._lock = None

        # positional parameters
        self.src_modalities = src_modalities
        self.tgt_modalities = tgt_modalities

        # training parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.batch_size_multiplier = batch_size_multiplier
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.gamma = gamma
        self.scale = scale
        self.criterion = criterion
        self.device = device
        self.verbose = verbose
        self._device_ids = _device_ids
        self._dataloader_num_workers = _dataloader_num_workers
        self._amp_enabled = _amp_enabled
        self._tmp_ckpt_filepath = _tmp_ckpt_filepath

    
    @_manage_ctx_fit
    def fit(self, x, y) -> Self:
        ''' ... '''
        # for PyTorch computational efficiency
        torch.set_num_threads(1)

        # initialize neural network
        self.net_ = self._init_net()

        # initialize dataloaders
        ldr_trn, ldr_vld = self._init_dataloader(x, y)

        # initialize optimizer and scheduler
        optimizer = self._init_optimizer()
        scheduler = self._init_scheduler(optimizer)

        # gradient scaler for AMP
        if self._amp_enabled: scaler = torch.cuda.amp.GradScaler()
        
        # initialize loss function (binary cross entropy)
        loss_func = self._init_loss_func({
            k: (
                sum([_[k] == 0 for _ in ldr_trn.dataset.tgt]),
                sum([_[k] == 1 for _ in ldr_trn.dataset.tgt]),
            ) for k in self.tgt_modalities
        })

        # to record the best validation performance criterion
        if self.criterion is not None: best_crit = None

        # progress bar for epoch loops
        if self.verbose == 1:
            with self._lock if self._lock is not None else suppress():
                pbr_epoch = tqdm.tqdm(
                    desc = 'Rank {:02d}'.format(self._rank),
                    total = self.num_epochs,
                    position = self._rank,
                    ascii = True,
                    leave = False,
                    bar_format='{l_bar}{r_bar}'
                )

        # training loop
        for epoch in range(self.num_epochs):
            # progress bar for batch loops
            if self.verbose > 1: 
                pbr_batch = ProgressBar(len(ldr_trn.dataset), 'Epoch {:03d} (TRN)'.format(epoch))

            # set model to train mode
            torch.set_grad_enabled(True)
            self.net_.train()

            scores_trn: dict[str, list[float]] = {k: [] for k in self.tgt_modalities}
            y_true_trn: dict[str, list[int]]   = {k: [] for k in self.tgt_modalities}
            losses_trn: dict[str, list[float]] = {k: [] for k in self.tgt_modalities}
            for n_iter, (x_batch, y_batch, _, mask_y) in enumerate(ldr_trn):
                # mount data to the proper device
                x_batch = {k: x_batch[k].to(self.device) for k in self.src_modalities}
                y_batch = {k: y_batch[k].to(torch.float).to(self.device) for k in self.tgt_modalities}
                # mask_x = {k: mask_x[k].to(self.device) for k in self.src_modalities}
                mask_y = {k: mask_y[k].to(self.device) for k in self.tgt_modalities}

                # forward
                with torch.autocast(
                    device_type = 'cpu' if self.device == 'cpu' else 'cuda',
                    dtype = torch.bfloat16 if self.device == 'cpu' else torch.float16,
                    enabled = self._amp_enabled,
                ):
                    outputs = self.net_(x_batch)

                    # calculate multitask loss
                    loss = 0
                    for i, tgt_k in enumerate(self.tgt_modalities):
                        loss_k = loss_func[tgt_k](outputs[tgt_k], y_batch[tgt_k])
                        loss_k = torch.masked_select(loss_k, torch.logical_not(mask_y[tgt_k].squeeze()))
                        loss += loss_k.mean()
                        losses_trn[tgt_k] += loss_k.detach().cpu().numpy().tolist()
                
                # backward
                if self._amp_enabled:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # update parameters
                if n_iter != 0 and n_iter % self.batch_size_multiplier == 0:
                    if self._amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                    else: 
                        optimizer.step()
                        optimizer.zero_grad()

                # save outputs to evaluate performance later
                for tgt_k in self.tgt_modalities:
                    tmp = torch.masked_select(outputs[tgt_k], torch.logical_not(mask_y[tgt_k].squeeze()))
                    scores_trn[tgt_k] += tmp.detach().cpu().numpy().tolist()
                    tmp = torch.masked_select(y_batch[tgt_k], torch.logical_not(mask_y[tgt_k].squeeze()))
                    y_true_trn[tgt_k] += tmp.cpu().numpy().tolist()

                # update progress bar
                if self.verbose > 1:
                    batch_size = len(next(iter(x_batch.values())))
                    pbr_batch.update(batch_size, {})
                    pbr_batch.refresh()

            # for better tqdm progress bar display
            if self.verbose > 1:
                pbr_batch.close()

            # set scheduler
            scheduler.step()

            # calculate and print training performance metrics
            y_pred_trn: dict[str, list[int]]   = {k: [] for k in self.tgt_modalities}
            y_prob_trn: dict[str, list[float]] = {k: [] for k in self.tgt_modalities}
            for tgt_k in self.tgt_modalities:
                for i in range(len(scores_trn[tgt_k])):
                    y_pred_trn[tgt_k].append(1 if scores_trn[tgt_k][i] > 0 else 0)
                    y_prob_trn[tgt_k].append(expit(scores_trn[tgt_k][i]))
            met_trn = get_metrics_multitask(y_true_trn, y_pred_trn, y_prob_trn)

            # add loss to metrics
            for tgt_k in self.tgt_modalities:
                met_trn[tgt_k]['Loss'] = np.mean(losses_trn[tgt_k])

            if self.verbose > 2:
                print_metrics_multitask(met_trn)

            # progress bar for validation
            if self.verbose > 1:
                pbr_batch = ProgressBar(len(ldr_vld.dataset), 'Epoch {:03d} (VLD)'.format(epoch))

            # set model to validation mode
            torch.set_grad_enabled(False)
            self.net_.eval()

            scores_vld: dict[str, list[float]] = {k: [] for k in self.tgt_modalities}
            y_true_vld: dict[str, list[int]]   = {k: [] for k in self.tgt_modalities}
            losses_vld: dict[str, list[float]] = {k: [] for k in self.tgt_modalities}
            for x_batch, y_batch, _, mask_y in ldr_vld:
                # mount data to the proper device
                x_batch = {k: x_batch[k].to(self.device) for k in self.src_modalities}
                y_batch = {k: y_batch[k].to(torch.float).to(self.device) for k in self.tgt_modalities}
                # mask_x = {k: mask_x[k].to(self.device) for k in self.src_modalities}
                mask_y = {k: mask_y[k].to(self.device) for k in self.tgt_modalities}

                # forward
                with torch.autocast(
                    device_type = 'cpu' if self.device == 'cpu' else 'cuda',
                    dtype = torch.bfloat16 if self.device == 'cpu' else torch.float16,
                    enabled = self._amp_enabled
                ):
                    outputs = self.net_(x_batch)

                    # calculate multitask loss
                    for i, tgt_k in enumerate(self.tgt_modalities):
                        loss_k = loss_func[tgt_k](outputs[tgt_k], y_batch[tgt_k])
                        loss_k = torch.masked_select(loss_k, torch.logical_not(mask_y[tgt_k].squeeze()))
                        losses_vld[tgt_k] += loss_k.detach().cpu().numpy().tolist()

                # save outputs to evaluate performance later
                for tgt_k in self.tgt_modalities:
                    tmp = torch.masked_select(outputs[tgt_k], torch.logical_not(mask_y[tgt_k].squeeze()))
                    scores_vld[tgt_k] += tmp.detach().cpu().numpy().tolist()
                    tmp = torch.masked_select(y_batch[tgt_k], torch.logical_not(mask_y[tgt_k].squeeze()))
                    y_true_vld[tgt_k] += tmp.cpu().numpy().tolist()

                # update progress bar
                if self.verbose > 1:
                    batch_size = len(next(iter(x_batch.values())))
                    pbr_batch.update(batch_size, {})
                    pbr_batch.refresh()

            # for better tqdm progress bar display
            if self.verbose > 1:
                pbr_batch.close()

            # calculate and print validation performance metrics
            y_pred_vld: dict[str, list[int]]   = {k: [] for k in self.tgt_modalities}
            y_prob_vld: dict[str, list[float]] = {k: [] for k in self.tgt_modalities}
            for tgt_k in self.tgt_modalities:
                for i in range(len(scores_vld[tgt_k])):
                    y_pred_vld[tgt_k].append(1 if scores_vld[tgt_k][i] > 0 else 0)
                    y_prob_vld[tgt_k].append(expit(scores_vld[tgt_k][i]))
            met_vld = get_metrics_multitask(y_true_vld, y_pred_vld, y_prob_vld)

            # add loss to metrics
            for tgt_k in self.tgt_modalities:
                met_vld[tgt_k]['Loss'] = np.mean(losses_vld[tgt_k])

            if self.verbose > 2:
                print_metrics_multitask(met_vld)

            # save the model if it has the best validation performance criterion by far
            if self.criterion is None: continue
            
            # is current criterion better than previous best?
            curr_crit = np.mean([met_vld[k][self.criterion] for k in self.tgt_modalities])
            if best_crit is None or np.isnan(best_crit):
                is_better = True
            elif self.criterion == 'Loss' and best_crit >= curr_crit:
                is_better = True
            elif self.criterion != 'Loss' and best_crit <= curr_crit:
                is_better = True
            else:
                is_better = False

            # update best criterion
            if is_better:
                best_crit = curr_crit
                best_state_dict = deepcopy(self.net_.state_dict())

                if self._tmp_ckpt_filepath is not None:
                    self.save(self._tmp_ckpt_filepath)

            if self.verbose > 2:
                print('Best {}: {}'.format(self.criterion, best_crit))

            if self.verbose == 1:
                with self._lock if self._lock is not None else suppress():
                    pbr_epoch.update(1)
                    pbr_epoch.refresh()

        if self.verbose == 1:
            with self._lock if self._lock is not None else suppress():
                pbr_epoch.close()

        # restore the model of the best validation performance across all epoches
        if ldr_vld is not None and self.criterion is not None:
            self.net_.load_state_dict(best_state_dict)

        return self

    def predict_logits(self,
        x: list[dict[str, Any]],
        _batch_size: int | None = None,
    ) -> list[dict[str, float]]:
        """
        The input x can be a single sample or a list of samples.
        """
        # input validation
        check_is_fitted(self)
        
        # for PyTorch computational efficiency
        torch.set_num_threads(1)

        # set model to eval mode
        torch.set_grad_enabled(False)
        self.net_.eval()

        # intialize dataset and dataloader object
        dat = TransformerTestingDataset(x, self.src_modalities)
        ldr = DataLoader(
            dataset = dat,
            batch_size = _batch_size if _batch_size is not None else len(x),
            shuffle = False,
            drop_last = False,
            num_workers = 0,
            collate_fn = TransformerTestingDataset.collate_fn,
        )

        # run model and collect results
        logits: list[dict[str, float]] = []
        for x_batch, _ in ldr:
            # mount data to the proper device
            x_batch = {k: x_batch[k].to(self.device) for k in self.src_modalities}

            # forward
            output: dict[str, Tensor] = self.net_(x_batch)
            
            # convert output from dict-of-list to list of dict, then append
            tmp = {k: output[k].tolist() for k in self.tgt_modalities}
            tmp = [{k: tmp[k][i] for k in self.tgt_modalities} for i in range(len(next(iter(tmp.values()))))]
            logits += tmp

        return logits
        
    def predict_proba(self,
        x: list[dict[str, Any]],
        temperature: float = 1.0,
        _batch_size: int | None = None,
    ) -> list[dict[str, float]]:
        ''' ... '''
        logits = self.predict_logits(x, _batch_size)
        return [{k: expit(smp[k] / temperature) for k in self.tgt_modalities} for smp in logits]

    def predict(self,
        x: list[dict[str, Any]],
        _batch_size: int | None = None,
    ) -> list[dict[str, int]]:
        ''' ... '''
        logits = self.predict_logits(x, _batch_size)
        return [{k: int(smp[k] > 0.0) for k in self.tgt_modalities} for smp in logits]

    def save(self, filepath: str) -> None:
        ''' ... '''
        check_is_fitted(self)
        state_dict = self.net_.state_dict()

        # attach model hyper parameters
        state_dict['src_modalities'] = self.src_modalities
        state_dict['tgt_modalities'] = self.tgt_modalities
        print('Saving model checkpoint to {} ... '.format(filepath), end='')
        torch.save(state_dict, filepath)
        print('Done.')

    def load(self, filepath: str) -> None:
        ''' ... '''
        # load state_dict
        state_dict = torch.load(filepath, map_location='cpu')

        # load essential parameters
        self.src_modalities: dict[str, dict[str, Any]] = state_dict.pop('src_modalities')
        self.tgt_modalities: dict[str, dict[str, Any]] = state_dict.pop('tgt_modalities')

        # initialize model
        self.net_ = nn.CNNResNet3DWithLinearClassifier(
            self.src_modalities,
            self.tgt_modalities,
        )

        # load model parameters
        self.net_.load_state_dict(state_dict)
        self.to(self.device)

    def to(self, device: str) -> Self:
        ''' Mount model to the given device. '''
        self.device = device
        if hasattr(self, 'net_'): self.net_ = self.net_.to(device)
        return self
    
    @classmethod
    def from_ckpt(cls, filepath: str) -> Self:
        ''' ... '''
        obj = cls(None, None)
        obj.load(filepath)
        return obj

    def _init_net(self):
        """ ... """
        net = nn.CNNResNet3DWithLinearClassifier(
            self.src_modalities,
            self.tgt_modalities,
        ).to(self.device)

        # train on multiple GPUs using torch.nn.DataParallel
        if self._device_ids is not None:
            net = torch.nn.DataParallel(net, device_ids=self._device_ids)

        # intialize model parameters using xavier_uniform
        for p in net.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        
        return net

    def _init_dataloader(self, x, y):
        """ ... """
        # split dataset
        x_trn, x_vld, y_trn, y_vld = train_test_split(
            x, y, test_size = 0.2, random_state = 0,
        )

        # initialize dataset and dataloader
        # dat_trn = TransformerTrainingDataset(
        dat_trn = Transformer2ndOrderBalancedTrainingDataset(
            x_trn, y_trn,
            self.src_modalities,
            self.tgt_modalities,
            dropout_rate = .5,
            # dropout_strategy = 'compensated',
            dropout_strategy = 'permutation',
        )

        dat_vld = TransformerValidationDataset(
            x_vld, y_vld,
            self.src_modalities,
            self.tgt_modalities,
        )

        ldr_trn = DataLoader(
            dataset = dat_trn,
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = False,
            num_workers = self._dataloader_num_workers,
            collate_fn = TransformerTrainingDataset.collate_fn,
            # pin_memory = True
        )

        ldr_vld = DataLoader(
            dataset = dat_vld,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = self._dataloader_num_workers,
            collate_fn = TransformerValidationDataset.collate_fn,
            # pin_memory = True
        )

        return ldr_trn, ldr_vld
    
    def _init_optimizer(self):
        """ ... """
        return torch.optim.AdamW(
            self.net_.parameters(),
            lr = self.lr,
            betas = (0.9, 0.98),
            weight_decay = self.weight_decay
        )
    
    def _init_scheduler(self, optimizer):
        """ ... """
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer = optimizer, 
            max_lr = self.lr,
            total_steps = self.num_epochs,
            verbose = (self.verbose > 2)
        )
    
    def _init_loss_func(self, 
        num_per_cls: dict[str, tuple[int, int]],
    ) -> dict[str, Module]:
        """ ... """
        return {k: nn.SigmoidFocalLoss(
            beta = self.beta,
            gamma = self.gamma,
            scale = self.scale,
            num_per_cls = num_per_cls[k],
            reduction = 'none',
        ) for k in self.tgt_modalities}