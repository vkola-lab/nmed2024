__all__ = ['ADRDModel']

import wandb
import torch
from torch.utils.data import DataLoader
import numpy as np
import tqdm
import multiprocessing
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from scipy.special import expit
from copy import deepcopy
from contextlib import suppress
from typing import Any, Self, Type
from functools import wraps
from tqdm import tqdm
Tensor = Type[torch.Tensor]
Module = Type[torch.nn.Module]

# for DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from .. import nn
from ..nn import Transformer
from ..utils import TransformerTrainingDataset, TransformerBalancedTrainingDataset, TransformerValidationDataset, TransformerTestingDataset, Transformer2ndOrderBalancedTrainingDataset
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
            kwargs['self'].to(default_device)
            return rtn
    return wrapper


class ADRDModel(BaseEstimator):
    """Primary model class for ADRD framework.

    The ADRDModel encapsulates the core pipeline of the ADRD framework, 
    permitting users to train and validate with the provided data. Designed for 
    user-friendly operation, the ADRDModel is derived from 
    ``sklearn.base.BaseEstimator``, ensuring compliance with the well-established 
    API design conventions of scikit-learn.
    """    
    def __init__(self,
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]],
        label_fractions: dict[str, float],
        d_model: int = 32,
        nhead: int = 1,
        num_encoder_layers: int = 1,
        num_decoder_layers: int = 1,
        num_epochs: int = 32,
        batch_size: int = 8,
        batch_size_multiplier: int = 1,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        beta: float = 0.9999,
        gamma: float = 2.0,
        criterion: str | None = None,
        device: str = 'cpu',
        cuda_devices: list = [1],
        img_net: str | None = None,
        imgnet_layers: int | None = 2,
        img_size: int | None = 128,
        fusion_stage: str = 'middle',
        patch_size: int | None = 16,
        imgnet_ckpt: str | None = None,
        train_imgnet: bool = False,
        ckpt_path: str = './adrd_tool/adrd/dev/ckpt/ckpt.pt',
        load_from_ckpt: bool = False,
        save_intermediate_ckpts: bool = False,
        data_parallel: bool = False,
        verbose: int = 0,
        wandb_ = 0,
        balanced_sampling: bool = False,
        label_distribution: dict = {},
        ranking_loss: bool = False,
        _device_ids: list | None = None,

        _dataloader_num_workers: int = 4,
        _amp_enabled: bool = False,
    ) -> None:  
        """Create a new ADRD model.

        :param src_modalities: _description_
        :type src_modalities: dict[str, dict[str, Any]]
        :param tgt_modalities: _description_
        :type tgt_modalities: dict[str, dict[str, Any]]
        :param label_fractions: _description_
        :type label_fractions: dict[str, float]
        :param d_model: _description_, defaults to 32
        :type d_model: int, optional
        :param nhead: number of transformer heads, defaults to 1
        :type nhead: int, optional
        :param num_encoder_layers: number of encoder layers, defaults to 1
        :type num_encoder_layers: int, optional
        :param num_decoder_layers: number of decoder layers, defaults to 1
        :type num_decoder_layers: int, optional
        :param num_epochs: number of training epochs, defaults to 32
        :type num_epochs: int, optional
        :param batch_size: batch size, defaults to 8
        :type batch_size: int, optional
        :param batch_size_multiplier: _description_, defaults to 1
        :type batch_size_multiplier: int, optional
        :param lr: learning rate, defaults to 1e-2
        :type lr: float, optional
        :param weight_decay: _description_, defaults to 0.0
        :type weight_decay: float, optional
        :param beta: _description_, defaults to 0.9999
        :type beta: float, optional
        :param gamma: The focusing parameter for the focal loss. Higher values of gamma make easy-to-classify examples contribute less to the loss relative to hard-to-classify examples. Must be non-negative., defaults to 2.0
        :type gamma: float, optional
        :param criterion: The criterion to select the best model, defaults to None
        :type criterion: str | None, optional
        :param device: 'cuda' or 'cpu', defaults to 'cpu'
        :type device: str, optional
        :param cuda_devices: A list of gpu numbers to data parallel training. The device must be set to 'cuda' and data_parallel must be set to True, defaults to [1]
        :type cuda_devices: list, optional
        :param img_net: _description_, defaults to None
        :type img_net: str | None, optional
        :param imgnet_layers: _description_, defaults to 2
        :type imgnet_layers: int | None, optional
        :param img_size: _description_, defaults to 128
        :type img_size: int | None, optional
        :param fusion_stage: _description_, defaults to 'middle'
        :type fusion_stage: str, optional
        :param patch_size: _description_, defaults to 16
        :type patch_size: int | None, optional
        :param imgnet_ckpt: _description_, defaults to None
        :type imgnet_ckpt: str | None, optional
        :param train_imgnet: Set to True to finetune the img_net backbone, defaults to False
        :type train_imgnet: bool, optional
        :param ckpt_path: The model checkpoint point path, defaults to './adrd_tool/adrd/dev/ckpt/ckpt.pt'
        :type ckpt_path: str, optional
        :param load_from_ckpt: Set to True to load the model weights from checkpoint ckpt_path, defaults to False
        :type load_from_ckpt: bool, optional
        :param save_intermediate_ckpts: Set to True to save intermediate model checkpoints, defaults to False
        :type save_intermediate_ckpts: bool, optional
        :param data_parallel: Set to True to enable data parallel trsining, defaults to False
        :type data_parallel: bool, optional
        :param verbose: _description_, defaults to 0
        :type verbose: int, optional
        :param wandb_: Set to 1 to track the loss and accuracy curves on wandb, defaults to 0
        :type wandb_: int, optional
        :param balanced_sampling: _description_, defaults to False
        :type balanced_sampling: bool, optional
        :param label_distribution: _description_, defaults to {}
        :type label_distribution: dict, optional
        :param ranking_loss: _description_, defaults to False
        :type ranking_loss: bool, optional
        :param _device_ids: _description_, defaults to None
        :type _device_ids: list | None, optional
        :param _dataloader_num_workers: _description_, defaults to 4
        :type _dataloader_num_workers: int, optional
        :param _amp_enabled: _description_, defaults to False
        :type _amp_enabled: bool, optional
        """
        # for multiprocessing
        self._rank = 0
        self._lock = None

        # positional parameters
        self.src_modalities = src_modalities
        self.tgt_modalities = tgt_modalities

        # training parameters
        self.label_fractions = label_fractions
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.batch_size_multiplier = batch_size_multiplier
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.gamma = gamma
        self.criterion = criterion
        self.device = device
        self.cuda_devices = cuda_devices
        self.img_net = img_net
        self.patch_size = patch_size
        self.img_size = img_size
        self.fusion_stage = fusion_stage
        self.imgnet_ckpt = imgnet_ckpt
        self.imgnet_layers = imgnet_layers
        self.train_imgnet = train_imgnet
        self.ckpt_path = ckpt_path
        self.load_from_ckpt = load_from_ckpt
        self.save_intermediate_ckpts = save_intermediate_ckpts
        self.data_parallel = data_parallel
        self.verbose = verbose
        self.label_distribution = label_distribution
        self.wandb_ = wandb_
        self.balanced_sampling = balanced_sampling
        self.ranking_loss = ranking_loss
        self._device_ids = _device_ids
        self._dataloader_num_workers = _dataloader_num_workers
        self._amp_enabled = _amp_enabled
        self.scaler = torch.cuda.amp.GradScaler()
        # self._init_net()

    @_manage_ctx_fit
    def fit(self, x_trn, x_vld, y_trn, y_vld, img_train_trans=None, img_vld_trans=None, img_mode=0) -> Self:
    # def fit(self, x, y) -> Self:
        ''' ... '''
        
        # start a new wandb run to track this script
        if self.wandb_ == 1:
            wandb.init(
                # set the wandb project where this run will be logged
                project="ADRD_main",
                
                # track hyperparameters and run metadata
                config={
                "Loss": 'Focalloss',
                "ranking_loss": self.ranking_loss,
                "img architecture": self.img_net,
                "EMB": "ALL_SEQ",
                "epochs": self.num_epochs,
                "d_model": self.d_model,
                # 'positional encoding': 'Diff PE',
                'Balanced Sampling': self.balanced_sampling,
                'Shared CNN': 'Yes',
                }
            )
            wandb.run.log_code(".")
        else:
            wandb.init(mode="disabled") 
        # for PyTorch computational efficiency
        torch.set_num_threads(1)
        # print(img_train_trans)
        # initialize neural network
        print(self.criterion)
        print(f"Ranking loss: {self.ranking_loss}")
        print(f"Batch size: {self.batch_size}")
        print(f"Batch size multiplier: {self.batch_size_multiplier}")

        if img_mode in [0,1,2]:
            for k, info in self.src_modalities.items():
                if info['type'] == 'imaging':
                    if 'densenet' in self.img_net.lower() and 'emb' not in self.img_net.lower():
                        info['shape'] = (1,) + self.img_size
                        info['img_shape'] = (1,) + self.img_size
                    elif 'emb' not in self.img_net.lower():
                        info['shape'] = (1,) + (self.img_size,) * 3
                        info['img_shape'] = (1,) + (self.img_size,) * 3
                    elif 'swinunetr' in self.img_net.lower():
                        info['shape'] = (1, 768, 4, 4, 4)
                        info['img_shape'] = (1, 768, 4, 4, 4)
        
        
        
        self._init_net()
        ldr_trn, ldr_vld = self._init_dataloader(x_trn, x_vld, y_trn, y_vld, img_train_trans=img_train_trans, img_vld_trans=img_vld_trans)

        # initialize optimizer and scheduler
        if not self.load_from_ckpt:
            self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler(self.optimizer)

        # gradient scaler for AMP
        if self._amp_enabled: 
            self.scaler = torch.cuda.amp.GradScaler()

        # initialize the focal losses 
        self.loss_fn = {}

        for k in self.tgt_modalities:
            if self.label_fractions[k] >= 0.3:
                alpha = -1
            else:
                alpha = pow((1 - self.label_fractions[k]), 2)
            # alpha = -1
            self.loss_fn[k] = nn.SigmoidFocalLoss(
                alpha = alpha,
                gamma = self.gamma,
                reduction = 'none'
            )
            

        # to record the best validation performance criterion
        if self.criterion is not None:
            best_crit = None
            best_crit_AUPR = None

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

        self.skip_embedding = {}
        for k, info in self.src_modalities.items():
            self.skip_embedding[k] = False

        self.grad_list = []
        # Define a hook function to print and store the gradient of a layer
        def print_and_store_grad(grad):
            self.grad_list.append(grad)
            # print(grad)
            
       
        # initialize the ranking loss
        self.lambda_coeff = 0.005
        self.margin = 0.25
        self.margin_loss = torch.nn.MarginRankingLoss(reduction='sum', margin=self.margin)
        
        # training loop
        for epoch in range(self.start_epoch, self.num_epochs):
            met_trn = self.train_one_epoch(ldr_trn, epoch)
            met_vld = self.validate_one_epoch(ldr_vld, epoch)
            
            print(self.ckpt_path.split('/')[-1])

            # save the model if it has the best validation performance criterion by far
            if self.criterion is None: continue
            
            # is current criterion better than previous best?
            curr_crit = np.mean([met_vld[i][self.criterion] for i in range(len(self.tgt_modalities))])
            curr_crit_AUPR = np.mean([met_vld[i]["AUC (PR)"] for i in range(len(self.tgt_modalities))])
            # AUROC
            if best_crit is None or np.isnan(best_crit):
                is_better = True
            elif self.criterion == 'Loss' and best_crit >= curr_crit:
                is_better = True
            elif self.criterion != 'Loss' and best_crit <= curr_crit :
                is_better = True
            else:
                is_better = False

            # AUPR
            if best_crit_AUPR is None or np.isnan(best_crit_AUPR):
                is_better_AUPR = True
            elif best_crit_AUPR <= curr_crit_AUPR :
                is_better_AUPR = True
            else:
                is_better_AUPR = False
            # update best criterion
            if is_better_AUPR:
                best_crit_AUPR = curr_crit_AUPR
                if self.save_intermediate_ckpts:
                    print(f"Saving the model to {self.ckpt_path[:-3]}_AUPR.pt...")
                    self.save(self.ckpt_path[:-3]+"_AUPR.pt", epoch)
            if is_better:
                best_crit = curr_crit
                best_state_dict = deepcopy(self.net_.state_dict())
                if self.save_intermediate_ckpts:
                    print(f"Saving the model to {self.ckpt_path}...")
                    self.save(self.ckpt_path, epoch)

            if self.verbose > 2:
                print('Best {}: {}'.format(self.criterion, best_crit))
                print('Best {}: {}'.format('AUC (PR)', best_crit_AUPR))

            if self.verbose == 1:
                with self._lock if self._lock is not None else suppress():
                    pbr_epoch.update(1)
                    pbr_epoch.refresh()

        if self.verbose == 1:
            with self._lock if self._lock is not None else suppress():
                pbr_epoch.close()

        return self
    
    def train_one_epoch(self, ldr_trn, epoch):
        # progress bar for batch loops
        if self.verbose > 1: 
            pbr_batch = ProgressBar(len(ldr_trn.dataset), 'Epoch {:03d} (TRN)'.format(epoch))

        # set model to train mode
        torch.set_grad_enabled(True)
        self.net_.train()

        scores_trn, y_true_trn, y_mask_trn = [], [], []
        losses_trn = [[] for _ in self.tgt_modalities]
        iters = len(ldr_trn)
        for n_iter, (x_batch, y_batch, mask, y_mask) in enumerate(ldr_trn):
           
            # mount data to the proper device
            x_batch = {k: x_batch[k].to(self.device) for k in x_batch}
            y_batch = {k: y_batch[k].to(torch.float).to(self.device) for k in y_batch}
            mask = {k: mask[k].to(self.device) for k in mask}
            y_mask = {k: y_mask[k].to(self.device) for k in y_mask}
            
            with torch.autocast(
                device_type = 'cpu' if self.device == 'cpu' else 'cuda',
                dtype = torch.bfloat16 if self.device == 'cpu' else torch.float16,
                enabled = self._amp_enabled,
            ):

                outputs = self.net_(x_batch, mask, skip_embedding=self.skip_embedding)
                
                # calculate multitask loss
                loss = 0

                # for initial 10 epochs, only the focal loss is used for stable training
                if self.ranking_loss:
                    if epoch < 10:
                        loss = 0
                    else:
                        for i, k in enumerate(self.tgt_modalities):
                           for ii, kk in enumerate(self.tgt_modalities):
                               if ii>i:
                                   pairs = (y_mask[k] == 1) & (y_mask[kk] == 1)
                                   total_elements = (torch.abs(y_batch[k][pairs]-y_batch[kk][pairs])).sum()
                                   if total_elements != 0:
                                       loss += self.lambda_coeff * (self.margin_loss(torch.sigmoid(outputs[k])[pairs],torch.sigmoid(outputs[kk][pairs]),y_batch[k][pairs]-y_batch[kk][pairs]))/total_elements

                for i, k in enumerate(self.tgt_modalities):
                    loss_task = self.loss_fn[k](outputs[k], y_batch[k])
                    msk_loss_task = loss_task * y_mask[k]
                    msk_loss_mean = msk_loss_task.sum() / y_mask[k].sum()
                    # msk_loss_mean = msk_loss_task.sum()
                    loss += msk_loss_mean
                    losses_trn[i] += msk_loss_task.detach().cpu().numpy().tolist()

            # backward
            loss = loss / self.batch_size_multiplier
            if self._amp_enabled:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if len(self.grad_list) > 0:
                print(len(self.grad_list), len(self.grad_list[-1]))
                print(f"Gradient at {n_iter}: {self.grad_list[-1][0]}")
           
            # print("img_MRI_T1_1 ", self.net_.modules_emb_src.img_MRI_T1_1.img_model.features[0].weight)
            # print("img_MRI_T1_1 ", self.net_.modules_emb_src.img_MRI_T1_1.downsample[0].weight)
            
            # update parameters
            if n_iter != 0 and n_iter % self.batch_size_multiplier == 0:
                if self._amp_enabled:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                else: 
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # set self.scheduler
                self.scheduler.step(epoch + n_iter / iters)

            ''' TODO: change array to dictionary later '''
            outputs = torch.stack(list(outputs.values()), dim=1)
            y_batch = torch.stack(list(y_batch.values()), dim=1)
            y_mask = torch.stack(list(y_mask.values()), dim=1)

            # save outputs to evaluate performance later
            scores_trn.append(outputs.detach().to(torch.float).cpu())
            y_true_trn.append(y_batch.cpu())
            y_mask_trn.append(y_mask.cpu())
            
            # update progress bar
            if self.verbose > 1:
                batch_size = len(next(iter(x_batch.values())))
                pbr_batch.update(batch_size, {})
                pbr_batch.refresh()

            # clear cuda cache
            if "cuda" in self.device:
                torch.cuda.empty_cache()

        # for better tqdm progress bar display
        if self.verbose > 1:
            pbr_batch.close()

        # calculate and print training performance metrics
        scores_trn = torch.cat(scores_trn)
        y_true_trn = torch.cat(y_true_trn)
        y_mask_trn = torch.cat(y_mask_trn)
        y_pred_trn = (scores_trn > 0).to(torch.int)
        y_prob_trn = torch.sigmoid(scores_trn)
        met_trn = get_metrics_multitask(
            y_true_trn.numpy(),
            y_pred_trn.numpy(),
            y_prob_trn.numpy(),
            y_mask_trn.numpy()
        )

        # add loss to metrics
        for i in range(len(self.tgt_modalities)):
            met_trn[i]['Loss'] = np.mean(losses_trn[i])
        
        # log metrics to wandb
        wandb.log({f"Train loss {list(self.tgt_modalities)[i]}": met_trn[i]['Loss']  for i in range(len(self.tgt_modalities))}, step=epoch)
        wandb.log({f"Train Balanced Accuracy {list(self.tgt_modalities)[i]}": met_trn[i]['Balanced Accuracy']  for i in range(len(self.tgt_modalities))}, step=epoch)
        
        wandb.log({f"Train AUC (ROC) {list(self.tgt_modalities)[i]}": met_trn[i]['AUC (ROC)']  for i in range(len(self.tgt_modalities))}, step=epoch)
        wandb.log({f"Train AUPR {list(self.tgt_modalities)[i]}": met_trn[i]['AUC (PR)']  for i in range(len(self.tgt_modalities))}, step=epoch)

        if self.verbose > 2:
            print_metrics_multitask(met_trn)
            
        return met_trn
    
    def validate_one_epoch(self, ldr_vld, epoch):
        # # progress bar for validation
        if self.verbose > 1:
            pbr_batch = ProgressBar(len(ldr_vld.dataset), 'Epoch {:03d} (VLD)'.format(epoch))

        # set model to validation mode
        torch.set_grad_enabled(False)
        self.net_.eval()

        scores_vld, y_true_vld, y_mask_vld = [], [], []
        losses_vld = [[] for _ in self.tgt_modalities]
        for x_batch, y_batch, mask, y_mask in ldr_vld:
            # if len(next(iter(x_batch.values()))) < self.batch_size:
            #     break
            # mount data to the proper device
            x_batch = {k: x_batch[k].to(self.device) for k in x_batch} # if 'img' not in k}
            # x_img_batch = {k: x_img_batch[k].to(self.device) for k in x_img_batch}
            y_batch = {k: y_batch[k].to(torch.float).to(self.device) for k in y_batch}
            mask = {k: mask[k].to(self.device) for k in mask}
            y_mask = {k: y_mask[k].to(self.device) for k in y_mask}

            # forward
            with torch.autocast(
                device_type = 'cpu' if self.device == 'cpu' else 'cuda',
                dtype = torch.bfloat16 if self.device == 'cpu' else torch.float16,
                enabled = self._amp_enabled
            ):
                
                outputs = self.net_(x_batch, mask, skip_embedding=self.skip_embedding)

                # calculate multitask loss
                for i, k in enumerate(self.tgt_modalities):
                    loss_task = self.loss_fn[k](outputs[k], y_batch[k])
                    msk_loss_task = loss_task * y_mask[k]
                    losses_vld[i] += msk_loss_task.detach().cpu().numpy().tolist()

            ''' TODO: change array to dictionary later '''
            outputs = torch.stack(list(outputs.values()), dim=1)
            y_batch = torch.stack(list(y_batch.values()), dim=1)
            y_mask = torch.stack(list(y_mask.values()), dim=1)

            # save outputs to evaluate performance later
            scores_vld.append(outputs.detach().to(torch.float).cpu())
            y_true_vld.append(y_batch.cpu())
            y_mask_vld.append(y_mask.cpu())

            # update progress bar
            if self.verbose > 1:
                batch_size = len(next(iter(x_batch.values())))
                pbr_batch.update(batch_size, {})
                pbr_batch.refresh()

            # clear cuda cache
            if "cuda" in self.device:
                torch.cuda.empty_cache()

        # for better tqdm progress bar display
        if self.verbose > 1:
            pbr_batch.close()

        # calculate and print validation performance metrics
        scores_vld = torch.cat(scores_vld)
        y_true_vld = torch.cat(y_true_vld)
        y_mask_vld = torch.cat(y_mask_vld)
        y_pred_vld = (scores_vld > 0).to(torch.int)
        y_prob_vld = torch.sigmoid(scores_vld)
        met_vld = get_metrics_multitask(
            y_true_vld.numpy(),
            y_pred_vld.numpy(),
            y_prob_vld.numpy(),
            y_mask_vld.numpy()
        )

        # add loss to metrics
        for i in range(len(self.tgt_modalities)):
            met_vld[i]['Loss'] = np.mean(losses_vld[i])
            
        wandb.log({f"Validation loss {list(self.tgt_modalities)[i]}": met_vld[i]['Loss'] for i in range(len(self.tgt_modalities))}, step=epoch)
        wandb.log({f"Validation Balanced Accuracy {list(self.tgt_modalities)[i]}": met_vld[i]['Balanced Accuracy']  for i in range(len(self.tgt_modalities))}, step=epoch)
        
        wandb.log({f"Validation AUC (ROC) {list(self.tgt_modalities)[i]}": met_vld[i]['AUC (ROC)']  for i in range(len(self.tgt_modalities))}, step=epoch)
        wandb.log({f"Validation AUPR {list(self.tgt_modalities)[i]}": met_vld[i]['AUC (PR)']  for i in range(len(self.tgt_modalities))}, step=epoch)

        if self.verbose > 2:
            print_metrics_multitask(met_vld)
        
        return met_vld
        

    def predict_logits(self,
        x: list[dict[str, Any]],
        _batch_size: int | None = None,
        skip_embedding: dict | None = None,
        img_transform: Any | None = None,
    ) -> list[dict[str, float]]:
        '''
        The input x can be a single sample or a list of samples.
        '''
        # input validation
        check_is_fitted(self)
        print(self.device)
        
        # for PyTorch computational efficiency
        torch.set_num_threads(1)

        # set model to eval mode
        torch.set_grad_enabled(False)
        self.net_.eval()

        # intialize dataset and dataloader object
        dat = TransformerTestingDataset(x, self.src_modalities, img_transform=img_transform)
        ldr = DataLoader(
            dataset = dat,
            batch_size = _batch_size if _batch_size is not None else len(x),
            shuffle = False,
            drop_last = False,
            num_workers = 0,
            collate_fn = TransformerTestingDataset.collate_fn,
        )
        # print("dataloader done")

        # run model and collect results
        logits: list[dict[str, float]] = []
        for x_batch, mask in tqdm(ldr):
            # mount data to the proper device
            # print(x_batch['his_SEX'])
            x_batch = {k: x_batch[k].to(self.device) for k in x_batch} 
            mask = {k: mask[k].to(self.device) for k in mask}

            # forward
            output: dict[str, Tensor] = self.net_(x_batch, mask, skip_embedding)
            
            # convert output from dict-of-list to list of dict, then append
            tmp = {k: output[k].tolist() for k in self.tgt_modalities}
            tmp = [{k: tmp[k][i] for k in self.tgt_modalities} for i in range(len(next(iter(tmp.values()))))]
            logits += tmp

        return logits
        
    def predict_proba(self,
        x: list[dict[str, Any]],
        skip_embedding: dict | None = None,
        temperature: float = 1.0,
        _batch_size: int | None = None,
        img_transform: Any | None = None,
    ) -> list[dict[str, float]]:
        ''' ... '''
        logits = self.predict_logits(x=x, _batch_size=_batch_size, img_transform=img_transform, skip_embedding=skip_embedding)
        print("got logits")
        return logits, [{k: expit(smp[k] / temperature) for k in self.tgt_modalities} for smp in logits]

    def predict(self,
        x: list[dict[str, Any]],
        skip_embedding: dict | None = None,
        fpr: dict[str, Any] | None = None,
        tpr: dict[str, Any] | None = None,
        thresholds: dict[str, Any] | None = None,
        _batch_size: int | None = None,
        img_transform: Any | None = None,
    ) -> list[dict[str, int]]:
        ''' ... '''
        if fpr is None or tpr is None or thresholds is None:
            logits, proba = self.predict_proba(x, _batch_size=_batch_size, img_transform=img_transform, skip_embedding=skip_embedding)
            print("got proba")
            return logits, proba, [{k: int(smp[k] > 0.5) for k in self.tgt_modalities} for smp in proba]
        else:
            logits, proba = self.predict_proba(x, _batch_size=_batch_size, img_transform=img_transform, skip_embedding=skip_embedding)
            print("got proba")
            youden_index = {}
            thr = {}
            for i, k in enumerate(self.tgt_modalities):
                youden_index[k] = tpr[i] - fpr[i]
                thr[k] = thresholds[i][np.argmax(youden_index[k])]
            #     print(thr[k])
            # print(thr)
            return logits, proba, [{k: int(smp[k] > thr[k]) for k in self.tgt_modalities} for smp in proba]

    def save(self, filepath: str, epoch: int) -> None:
        """Save the model to the given file stream.

        :param filepath: _description_
        :type filepath: str
        :param epoch: _description_
        :type epoch: int
        """        
        check_is_fitted(self)
        if self.data_parallel:
            state_dict = self.net_.module.state_dict()
        else:
            state_dict = self.net_.state_dict()

        # attach model hyper parameters
        state_dict['src_modalities'] = self.src_modalities
        state_dict['tgt_modalities'] = self.tgt_modalities
        state_dict['d_model'] = self.d_model
        state_dict['nhead'] = self.nhead
        state_dict['num_encoder_layers'] = self.num_encoder_layers
        state_dict['num_decoder_layers'] = self.num_decoder_layers
        state_dict['optimizer'] = self.optimizer
        state_dict['img_net'] = self.img_net
        state_dict['imgnet_layers'] = self.imgnet_layers
        state_dict['img_size'] = self.img_size
        state_dict['patch_size'] = self.patch_size
        state_dict['imgnet_ckpt'] = self.imgnet_ckpt
        state_dict['train_imgnet'] = self.train_imgnet
        state_dict['epoch'] = epoch

        if self.scaler is not None:
            state_dict['scaler'] = self.scaler.state_dict()
        if self.label_distribution:
            state_dict['label_distribution'] = self.label_distribution

        torch.save(state_dict, filepath)

    def load(self, filepath: str, map_location: str = 'cpu', img_dict=None) -> None:
        """Load a model from the given file stream.

        :param filepath: _description_
        :type filepath: str
        :param map_location: _description_, defaults to 'cpu'
        :type map_location: str, optional
        :param img_dict: _description_, defaults to None
        :type img_dict: _type_, optional
        """        
        # load state_dict
        state_dict = torch.load(filepath, map_location=map_location)

        # load data modalities
        self.src_modalities: dict[str, dict[str, Any]] = state_dict.pop('src_modalities')
        self.tgt_modalities: dict[str, dict[str, Any]] = state_dict.pop('tgt_modalities')
        if 'label_distribution' in state_dict:
            self.label_distribution: dict[str, dict[int, int]] = state_dict.pop('label_distribution')
        if 'optimizer' in state_dict:
            self.optimizer = state_dict.pop('optimizer')

        # initialize model
        self.d_model = state_dict.pop('d_model')
        self.nhead = state_dict.pop('nhead')
        self.num_encoder_layers = state_dict.pop('num_encoder_layers')
        self.num_decoder_layers = state_dict.pop('num_decoder_layers')
        if 'epoch' in state_dict.keys():
            self.start_epoch = state_dict.pop('epoch')
        if img_dict is None:
            self.img_net = state_dict.pop('img_net')
            self.imgnet_layers = state_dict.pop('imgnet_layers')
            self.img_size = state_dict.pop('img_size')
            self.patch_size = state_dict.pop('patch_size')
            self.imgnet_ckpt = state_dict.pop('imgnet_ckpt')
            self.train_imgnet = state_dict.pop('train_imgnet')
        else:
            self.img_net  = img_dict['img_net']
            self.imgnet_layers  = img_dict['imgnet_layers']
            self.img_size  = img_dict['img_size']
            self.patch_size  = img_dict['patch_size']
            self.imgnet_ckpt  = img_dict['imgnet_ckpt']
            self.train_imgnet  = img_dict['train_imgnet']
            state_dict.pop('img_net')
            state_dict.pop('imgnet_layers')
            state_dict.pop('img_size')
            state_dict.pop('patch_size')
            state_dict.pop('imgnet_ckpt')
            state_dict.pop('train_imgnet')
            
        for k, info in self.src_modalities.items():
            if info['type'] == 'imaging':
                if 'emb' not in self.img_net.lower():
                    info['shape'] = (1,) + (self.img_size,) * 3
                    info['img_shape'] = (1,) + (self.img_size,) * 3
                elif 'swinunetr' in self.img_net.lower():
                    info['shape'] = (1, 768, 4, 4, 4)
                    info['img_shape'] = (1, 768, 4, 4, 4)
                # print(info['shape'])

        self.net_ = Transformer(self.src_modalities, self.tgt_modalities, self.d_model, self.nhead, self.num_encoder_layers, self.num_decoder_layers, self.device, self.cuda_devices, self.img_net, self.imgnet_layers, self.img_size, self.patch_size, self.imgnet_ckpt, self.train_imgnet, self.fusion_stage)

       
        if 'scaler' in state_dict and state_dict['scaler']:
            self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.net_.load_state_dict(state_dict)
        check_is_fitted(self)
        self.net_.to(self.device)

    def to(self, device: str) -> Self:
        """Mount the model to the given device. 

        :param device: _description_
        :type device: str
        :return: _description_
        :rtype: Self
        """        
        self.device = device
        if hasattr(self, 'model'): self.net_ = self.net_.to(device)
        if hasattr(self, 'img_model'): self.img_model = self.img_model.to(device)
        return self
    
    @classmethod
    def from_ckpt(cls, filepath: str, device='cpu', img_dict=None) -> Self:
        """Create a new ADRD model and load parameters from the checkpoint. 

        This is an alternative constructor.

        :param filepath: _description_
        :type filepath: str
        :param device: _description_, defaults to 'cpu'
        :type device: str, optional
        :param img_dict: _description_, defaults to None
        :type img_dict: _type_, optional
        :return: _description_
        :rtype: Self
        """ 
        obj = cls(None, None, None,device=device)
        if device == 'cuda':
            obj.device = "{}:{}".format(obj.device, str(obj.cuda_devices[0]))
        print(obj.device)
        obj.load(filepath, map_location=obj.device, img_dict=img_dict)
        return obj
    
    def _init_net(self):
        """ ... """
        # set the device for use
        if self.device == 'cuda':
            self.device = "{}:{}".format(self.device, str(self.cuda_devices[0]))
        print("Device: " + self.device)
        
        self.start_epoch = 0
        if self.load_from_ckpt:
            try:
                print("Loading model from checkpoint...")
                self.load(self.ckpt_path, map_location=self.device)
            except:
                print("Cannot load from checkpoint. Initializing new model...")
                self.load_from_ckpt = False

        if not self.load_from_ckpt:
            self.net_ = nn.Transformer(
                src_modalities = self.src_modalities, 
                tgt_modalities = self.tgt_modalities, 
                d_model = self.d_model, 
                nhead = self.nhead, 
                num_encoder_layers = self.num_encoder_layers, 
                num_decoder_layers = self.num_decoder_layers, 
                device = self.device, 
                cuda_devices = self.cuda_devices, 
                img_net = self.img_net, 
                layers = self.imgnet_layers, 
                img_size = self.img_size, 
                patch_size = self.patch_size, 
                imgnet_ckpt = self.imgnet_ckpt, 
                train_imgnet = self.train_imgnet,
                fusion_stage = self.fusion_stage,
            )  
            
            # intialize model parameters using xavier_uniform
            for name, p in self.net_.named_parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
        
        self.net_.to(self.device)

        # Initialize the number of GPUs
        if self.data_parallel and torch.cuda.device_count() > 1:
            print("Available", torch.cuda.device_count(), "GPUs!")
            self.net_ = torch.nn.DataParallel(self.net_, device_ids=self.cuda_devices)

        # return net

    def _init_dataloader(self, x_trn, x_vld, y_trn, y_vld, img_train_trans=None, img_vld_trans=None):    
        # initialize dataset and dataloader
        if self.balanced_sampling:
            dat_trn = Transformer2ndOrderBalancedTrainingDataset(
                x_trn, y_trn,
                self.src_modalities,
                self.tgt_modalities,
                dropout_rate = .5,
                dropout_strategy = 'permutation',
                img_transform=img_train_trans,
            )
        else:
            dat_trn = TransformerTrainingDataset(
                x_trn, y_trn,
                self.src_modalities,
                self.tgt_modalities,
                dropout_rate = .5,
                dropout_strategy = 'permutation',
                img_transform=img_train_trans,
            )

        dat_vld = TransformerValidationDataset(
            x_vld, y_vld,
            self.src_modalities,
            self.tgt_modalities,
            img_transform=img_vld_trans,
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
        params = list(self.net_.parameters())
        return torch.optim.AdamW(
            params,
            lr = self.lr,
            betas = (0.9, 0.98),
            weight_decay = self.weight_decay
        )
    
    def _init_scheduler(self, optimizer):
        """ ... """       

        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=64,
            T_mult=2,
            eta_min = 0,
            verbose=(self.verbose > 2)
        )
    
    def _init_loss_func(self, 
        num_per_cls: dict[str, tuple[int, int]],
    ) -> dict[str, Module]:
        """ ... """
        return {k: nn.SigmoidFocalLossBeta(
            beta = self.beta,
            gamma = self.gamma,
            num_per_cls = num_per_cls[k],
            reduction = 'none',
        ) for k in self.tgt_modalities}
    
    def _proc_fit(self):
        """ ... """
