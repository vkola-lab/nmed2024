__all__ = ['Transformer']

import wandb
import torch
import numpy as np
import functools
import inspect
import monai
import random

from tqdm import tqdm
from functools import wraps
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from scipy.special import expit
from copy import deepcopy
from contextlib import suppress
from typing import Any, Self, Type
Tensor = Type[torch.Tensor]
Module = Type[torch.nn.Module]
from torch.utils.data import DataLoader
from monai.utils.type_conversion import convert_to_tensor
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

# for DistributedDataParallel
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from .. import nn
from ..utils.misc import ProgressBar
from ..utils.misc import get_metrics_multitask, print_metrics_multitask
from ..utils.misc import convert_args_kwargs_to_kwargs

import warnings
warnings.filterwarnings("ignore")


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

def collate_handle_corrupted(samples_list, dataset, labels, dtype=torch.half):
    # print(len(samples_list))
    orig_len = len(samples_list)
    # for the loss to be consistent, we drop samples with NaN values in any of their corresponding crops
    for i, s in enumerate(samples_list):
        ic(s is None)
        if s is None:
            continue
    samples_list = list(filter(lambda x: x is not None, samples_list))

    if len(samples_list) == 0:
        ic('recursive call')
        return collate_handle_corrupted([dataset[random.randint(0, len(dataset)-1)] for _ in range(orig_len)], dataset, labels)
    
    # collated_images = torch.stack([convert_to_tensor(s["image"]) for s in samples_list])
    try:
        if "image" in samples_list[0]:
            samples_list = [s for s in samples_list if not torch.isnan(s["image"]).any()]
            # print('samples list: ', len(samples_list))
            collated_images = torch.stack([convert_to_tensor(s["image"]) for s in samples_list])
            # print("here1")
            collated_labels = {k: torch.Tensor([s["label"][k] if s["label"][k] is not None else 0 for s in samples_list]) for k in labels}
            # print("here2")
            collated_mask = {k: torch.Tensor([1 if s["label"][k] is not None else 0 for s in samples_list]) for k in labels}
            # print("here3")
            return {"image": collated_images,
                    "label": collated_labels,
                    "mask": collated_mask}
    except:
        return collate_handle_corrupted([dataset[random.randint(0, len(dataset)-1)] for _ in range(orig_len)], dataset, labels)
    
    
     
def get_backend(img_backend):
    if img_backend == 'C3D':
        return nn.C3D
    elif img_backend == 'DenseNet':
        return nn.DenseNet


class ImagingModel(BaseEstimator):
    ''' ... '''
    def __init__(self,
        tgt_modalities: list[str],
        label_fractions: dict[str, float],
        num_epochs: int = 32,
        batch_size: int = 8,
        batch_size_multiplier: int = 1,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        beta: float = 0.9999,
        gamma: float = 2.0,
        bn_size: int = 4,
        growth_rate: int = 12, 
        block_config: tuple = (3, 3, 3), 
        compression: float = 0.5,
        num_init_features: int = 16, 
        drop_rate: float = 0.2,
        criterion: str | None = None,
        device: str = 'cpu',
        cuda_devices: list = [2],
        ckpt_path: str = '/home/skowshik/ADRD_repo/adrd_tool/dev/ckpt/ckpt.pt',
        load_from_ckpt: bool = True,
        save_intermediate_ckpts: bool = False,
        data_parallel: bool = False,
        verbose: int = 0,
        img_backend: str | None = None,
        label_distribution: dict = {},
        wandb_ = 1,
        _device_ids: list | None = None,
        _dataloader_num_workers: int = 4,
        _amp_enabled: bool = False,
    ) -> None:  
        ''' ... '''
        # for multiprocessing
        self._rank = 0
        self._lock = None

        # positional parameters
        self.tgt_modalities = tgt_modalities

        # training parameters
        self.label_fractions = label_fractions
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.batch_size_multiplier = batch_size_multiplier
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta = beta
        self.gamma = gamma
        self.bn_size = bn_size
        self.growth_rate = growth_rate
        self.block_config = block_config
        self.compression = compression
        self.num_init_features = num_init_features
        self.drop_rate = drop_rate
        self.criterion = criterion
        self.device = device
        self.cuda_devices = cuda_devices
        self.ckpt_path = ckpt_path
        self.load_from_ckpt = load_from_ckpt
        self.save_intermediate_ckpts = save_intermediate_ckpts
        self.data_parallel = data_parallel
        self.verbose = verbose
        self.img_backend = img_backend
        self.label_distribution = label_distribution
        self.wandb_ = wandb_
        self._device_ids = _device_ids
        self._dataloader_num_workers = _dataloader_num_workers
        self._amp_enabled = _amp_enabled
        self.scaler = torch.cuda.amp.GradScaler()

    @_manage_ctx_fit
    def fit(self, trn_list, vld_list, img_train_trans=None, img_vld_trans=None) -> Self:
    # def fit(self, x, y) -> Self:
        ''' ... '''
        
        # start a new wandb run to track this script
        if self.wandb_ == 1:
            wandb.init(
                # set the wandb project where this run will be logged
                project="ADRD_main",
                
                # track hyperparameters and run metadata
                config={
                "Model": "DenseNet",
                "Loss": 'Focalloss',
                "EMB": "ALL_EMB",
                "epochs": 256,
                }
            )
            wandb.run.log_code("/home/skowshik/ADRD_repo/pipeline_v1_main/adrd_tool")
        else:
            wandb.init(mode="disabled") 
        # for PyTorch computational efficiency
        torch.set_num_threads(1)
        print(self.criterion)

        # initialize neural network
        self._init_net()

        # for k, info in self.src_modalities.items():
        #     if info['type'] == 'imaging' and self.img_net != 'EMB':
        #         info['shape'] = (1,) + (self.img_size,) * 3
        #         info['img_shape'] = (1,) + (self.img_size,) * 3
                # print(info['shape'])

        # initialize dataloaders
        # ldr_trn, ldr_vld = self._init_dataloader(x, y)
        # ldr_trn, ldr_vld = self._init_dataloader(x_trn, x_vld, y_trn, y_vld)
        ldr_trn, ldr_vld = self._init_dataloader(trn_list, vld_list, img_train_trans=img_train_trans, img_vld_trans=img_vld_trans)

        # initialize optimizer and scheduler
        if not self.load_from_ckpt:
            self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler(self.optimizer)

        # gradient scaler for AMP
        if self._amp_enabled: 
            self.scaler = torch.cuda.amp.GradScaler()

        # initialize focal loss function 
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
                pbr_epoch = tqdm(
                    desc = 'Rank {:02d}'.format(self._rank),
                    total = self.num_epochs,
                    position = self._rank,
                    ascii = True,
                    leave = False,
                    bar_format='{l_bar}{r_bar}'
                )

        # Define a hook function to print and store the gradient of a layer
        def print_and_store_grad(grad, grad_list):
            grad_list.append(grad)
            # print(grad)

        # grad_list = []
        # self.net_.modules_emb_src['img_MRI_T1'].downsample[0].weight.register_hook(lambda grad: print_and_store_grad(grad, grad_list))

        # lambda_coeff = 0.0001
        # margin_loss = torch.nn.MarginRankingLoss(reduction='sum', margin=0.05)
        
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

        return self
    
    def train_one_epoch(self, ldr_trn, epoch):
        
        # progress bar for batch loops
        if self.verbose > 1: 
            pbr_batch = ProgressBar(len(ldr_trn.dataset), 'Epoch {:03d} (TRN)'.format(epoch))
        
        torch.set_grad_enabled(True)
        self.net_.train()
        
        scores_trn, y_true_trn, y_mask_trn = [], [], []
        losses_trn = [[] for _ in self.tgt_modalities]
        iters = len(ldr_trn)
        print(iters)
        for n_iter, batch_data in enumerate(ldr_trn):
            # if len(batch_data["image"]) < self.batch_size:
            #     continue
            
            x_batch = batch_data["image"].to(self.device, non_blocking=True)
            y_batch = {k: v.to(self.device, non_blocking=True) for k,v in batch_data["label"].items()}
            y_mask = {k: v.to(self.device, non_blocking=True) for k,v in batch_data["mask"].items()}
            
            with torch.autocast(
                device_type = 'cpu' if self.device == 'cpu' else 'cuda',
                dtype = torch.bfloat16 if self.device == 'cpu' else torch.float16,
                enabled = self._amp_enabled,
            ):

                outputs = self.net_(x_batch, shap=False)
                # print(outputs.shape)
                # calculate multitask loss
                loss = 0
                for i, k in enumerate(self.tgt_modalities):
                    loss_task = self.loss_fn[k](outputs[k], y_batch[k])
                    msk_loss_task = loss_task * y_mask[k]
                    msk_loss_mean = msk_loss_task.sum() / y_mask[k].sum()
                    loss += msk_loss_mean
                    losses_trn[i] += msk_loss_task.detach().cpu().numpy().tolist()

            # backward
            if self._amp_enabled:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # print(len(grad_list), len(grad_list[-1]))
            # print(f"Gradient at {n_iter}: {grad_list[-1][0]}")
            
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
            # print(f"Weight: {self.net_.module.features[0].weight[0]}")

            ''' TODO: change array to dictionary later '''
            outputs = torch.stack(list(outputs.values()), dim=1)
            y_batch = torch.stack(list(y_batch.values()), dim=1)
            y_mask = torch.stack(list(y_mask.values()), dim=1)

            # save outputs to evaluate performance later
            scores_trn.append(outputs.detach().to(torch.float).cpu())
            y_true_trn.append(y_batch.cpu())
            y_mask_trn.append(y_mask.cpu())
            
            # log metrics to wandb

            # update progress bar
            if self.verbose > 1:
                batch_size = len(x_batch)
                pbr_batch.update(batch_size, {})
                pbr_batch.refresh()

            # clear cuda cache
            if "cuda" in self.device:
                torch.cuda.empty_cache()

        # for better tqdm progress bar display
        if self.verbose > 1:
            pbr_batch.close()

        # # set self.scheduler
        # self.scheduler.step()

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
        
        wandb.log({f"Train loss {list(self.tgt_modalities)[i]}": met_trn[i]['Loss']  for i in range(len(self.tgt_modalities))}, step=epoch)
        wandb.log({f"Train Balanced Accuracy {list(self.tgt_modalities)[i]}": met_trn[i]['Balanced Accuracy']  for i in range(len(self.tgt_modalities))}, step=epoch)
        
        wandb.log({f"Train AUC (ROC) {list(self.tgt_modalities)[i]}": met_trn[i]['AUC (ROC)']  for i in range(len(self.tgt_modalities))}, step=epoch)
        wandb.log({f"Train AUPR {list(self.tgt_modalities)[i]}": met_trn[i]['AUC (PR)']  for i in range(len(self.tgt_modalities))}, step=epoch)

        if self.verbose > 2:
            print_metrics_multitask(met_trn)
        
        return met_trn
    
    # @torch.no_grad()
    def validate_one_epoch(self, ldr_vld, epoch):
        # progress bar for validation
        if self.verbose > 1:
            pbr_batch = ProgressBar(len(ldr_vld.dataset), 'Epoch {:03d} (VLD)'.format(epoch))

        # set model to validation mode
        torch.set_grad_enabled(False)
        self.net_.eval()

        scores_vld, y_true_vld, y_mask_vld = [], [], []
        losses_vld = [[] for _ in self.tgt_modalities]
        for batch_data in ldr_vld:
            # if len(batch_data["image"]) < self.batch_size:
            #     continue
            x_batch = batch_data["image"].to(self.device, non_blocking=True)
            y_batch = {k: v.to(self.device, non_blocking=True) for k,v in batch_data["label"].items()}
            y_mask = {k: v.to(self.device, non_blocking=True) for k,v in batch_data["mask"].items()}
            
            # forward
            with torch.autocast(
                device_type = 'cpu' if self.device == 'cpu' else 'cuda',
                dtype = torch.bfloat16 if self.device == 'cpu' else torch.float16,
                enabled = self._amp_enabled
            ):
                
                outputs = self.net_(x_batch, shap=False)

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
                batch_size = len(x_batch)
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
    

    def save(self, filepath: str, epoch: int = 0) -> None:
        ''' ... '''
        check_is_fitted(self)
        if self.data_parallel:
            state_dict = self.net_.module.state_dict()
        else:
            state_dict = self.net_.state_dict()

        # attach model hyper parameters
        state_dict['tgt_modalities'] = self.tgt_modalities
        state_dict['optimizer'] = self.optimizer
        state_dict['bn_size'] = self.bn_size
        state_dict['growth_rate'] = self.growth_rate
        state_dict['block_config'] = self.block_config
        state_dict['compression'] = self.compression
        state_dict['num_init_features'] = self.num_init_features
        state_dict['drop_rate'] = self.drop_rate
        state_dict['epoch'] = epoch

        if self.scaler is not None:
            state_dict['scaler'] = self.scaler.state_dict()
        if self.label_distribution:
            state_dict['label_distribution'] = self.label_distribution

        torch.save(state_dict, filepath)

    def load(self, filepath: str, map_location: str = 'cpu', how='latest') -> None:
        ''' ... '''
        # load state_dict
        if how == 'latest':
            if torch.load(filepath)['epoch'] > torch.load(f'{filepath[:-3]}_AUPR.pt')['epoch']:
                print("Loading model saved using AUROC")
                state_dict = torch.load(filepath, map_location=map_location)
            else:
                print("Loading model saved using AUPR")
                state_dict = torch.load(f'{filepath[:-3]}_AUPR.pt', map_location=map_location)
        else:
            state_dict = torch.load(filepath, map_location=map_location)

        # load data modalities
        self.tgt_modalities: dict[str, dict[str, Any]] = state_dict.pop('tgt_modalities')
        if 'label_distribution' in state_dict:
            self.label_distribution: dict[str, dict[int, int]] = state_dict.pop('label_distribution')
        if 'optimizer' in state_dict:
            self.optimizer = state_dict.pop('optimizer')
        if 'bn_size' in state_dict:
            self.bn_size = state_dict.pop('bn_size')
        if 'growth_rate' in state_dict:
            self.growth_rate = state_dict.pop('growth_rate')
        if 'block_config' in state_dict:
            self.block_config = state_dict.pop('block_config')
        if 'compression' in state_dict:
            self.compression = state_dict.pop('compression')
        if 'num_init_features' in state_dict:
            self.num_init_features = state_dict.pop('num_init_features')
        if 'drop_rate' in state_dict:
            self.drop_rate = state_dict.pop('drop_rate')
        if 'epoch' in state_dict:
            self.start_epoch = state_dict.pop('epoch')
            print(f'Epoch: {self.start_epoch}')

        # initialize model

        self.net_ = get_backend(self.img_backend)( 
                tgt_modalities = self.tgt_modalities,
                bn_size = self.bn_size,
                growth_rate=self.growth_rate, 
                block_config=self.block_config, 
                compression=self.compression,
                num_init_features=self.num_init_features, 
                drop_rate=self.drop_rate,
                load_from_ckpt=self.load_from_ckpt
            )
        print(self.net_)
            
        if 'scaler' in state_dict and state_dict['scaler']:
            self.scaler.load_state_dict(state_dict.pop('scaler'))
        self.net_.load_state_dict(state_dict)
        check_is_fitted(self)
        self.net_.to(self.device)

    def to(self, device: str) -> Self:
        ''' Mount model to the given device. '''
        self.device = device
        if hasattr(self, 'model'): self.net_ = self.net_.to(device)
        return self
    
    @classmethod
    def from_ckpt(cls, filepath: str, device='cpu', img_backend=None, load_from_ckpt=True, how='latest') -> Self:
        ''' ... '''
        obj = cls(None, None, None,device=device)
        if device == 'cuda':
            obj.device = "{}:{}".format(obj.device, str(obj.cuda_devices[0]))
        print(obj.device)
        obj.img_backend=img_backend
        obj.load_from_ckpt = load_from_ckpt
        obj.load(filepath, map_location=obj.device, how=how)
        return obj
    
    def _init_net(self):
        """ ... """
        self.start_epoch = 0
        # set the device for use
        if self.device == 'cuda':
            self.device = "{}:{}".format(self.device, str(self.cuda_devices[0]))
        # self.load(self.ckpt_path, map_location=self.device)
        # print("Loading model from checkpoint...")
        # self.load(self.ckpt_path, map_location=self.device)

        if self.load_from_ckpt:
            try:
                print("Loading model from checkpoint...")
                self.load(self.ckpt_path, map_location=self.device)
            except:
                print("Cannot load from checkpoint. Initializing new model...")
                self.load_from_ckpt = False

        if not self.load_from_ckpt:
            self.net_ = get_backend(self.img_backend)( 
                tgt_modalities = self.tgt_modalities,
                bn_size = self.bn_size,
                growth_rate=self.growth_rate, 
                block_config=self.block_config, 
                compression=self.compression,
                num_init_features=self.num_init_features, 
                drop_rate=self.drop_rate,
                load_from_ckpt=self.load_from_ckpt
            ) 
            
            # # intialize model parameters using xavier_uniform
            # for p in self.net_.parameters():
            #     if p.dim() > 1:
            #         torch.nn.init.xavier_uniform_(p)
        
        self.net_.to(self.device)

        # Initialize the number of GPUs
        if self.data_parallel and torch.cuda.device_count() > 1:
            print("Available", torch.cuda.device_count(), "GPUs!")
            self.net_ = torch.nn.DataParallel(self.net_, device_ids=self.cuda_devices)

        # return net

    def _init_dataloader(self, trn_list, vld_list, img_train_trans=None, img_vld_trans=None):
    # def _init_dataloader(self, x, y):
        """ ... """
        # # split dataset
        # x_trn, x_vld, y_trn, y_vld = train_test_split(
        #     x, y, test_size = 0.2, random_state = 0,
        # )

        # # initialize dataset and dataloader
        # dat_trn = CNNTrainingValidationDataset(
        #     x_trn, y_trn,
        #     self.tgt_modalities,
        #     img_transform=img_train_trans,
        # )

        # dat_vld = CNNTrainingValidationDataset(
        #     x_vld, y_vld,
        #     self.tgt_modalities,
        #     img_transform=img_vld_trans,
        # )
        
        dat_trn = monai.data.Dataset(data=trn_list, transform=img_train_trans)
        dat_vld = monai.data.Dataset(data=vld_list, transform=img_vld_trans)
        collate_fn_trn = functools.partial(collate_handle_corrupted, dataset=dat_trn, dtype=torch.FloatTensor, labels=self.tgt_modalities)
        collate_fn_vld = functools.partial(collate_handle_corrupted, dataset=dat_vld, dtype=torch.FloatTensor, labels=self.tgt_modalities)

        ldr_trn = DataLoader(
            dataset = dat_trn,
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = False,
            num_workers = self._dataloader_num_workers,
            collate_fn = collate_fn_trn,
            # pin_memory = True
        )

        ldr_vld = DataLoader(
            dataset = dat_vld,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = self._dataloader_num_workers,
            collate_fn = collate_fn_vld,
            # pin_memory = True
        )

        return ldr_trn, ldr_vld
    
    def _init_optimizer(self):
        """ ... """
        params = list(self.net_.parameters()) 
        # for p in params:
        #     print(p.requires_grad)
        return torch.optim.AdamW(
            params,
            lr = self.lr,
            betas = (0.9, 0.98),
            weight_decay = self.weight_decay
        )
    
    def _init_scheduler(self, optimizer):
        """ ... """
        # return torch.optim.lr_scheduler.OneCycleLR(
        #     optimizer = optimizer, 
        #     max_lr = self.lr,
        #     total_steps = self.num_epochs,
        #     verbose = (self.verbose > 2)
        # )

        # return torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer=optimizer, 
        #     T_max=64, 
        #     verbose=(self.verbose > 2)
        # )

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
        
    def _init_test_dataloader(self, batch_size, tst_list, img_tst_trans=None):
        # input validation
        check_is_fitted(self)
        print(self.device)
        
        # for PyTorch computational efficiency
        torch.set_num_threads(1)

        # set model to eval mode
        torch.set_grad_enabled(False)
        self.net_.eval()
        
        dat_tst = monai.data.Dataset(data=tst_list, transform=img_tst_trans)
        collate_fn_tst = functools.partial(collate_handle_corrupted, dataset=dat_tst, dtype=torch.FloatTensor, labels=self.tgt_modalities)
        # print(collate_fn_tst)

        ldr_tst = DataLoader(
            dataset = dat_tst,
            batch_size = batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = self._dataloader_num_workers,
            collate_fn = collate_fn_tst,
            # pin_memory = True
        )
        return ldr_tst

    
    def predict_logits(self,
        ldr_tst: Any | None = None,
    ) -> list[dict[str, float]]:
        
        # run model and collect results
        logits: list[dict[str, float]] = []
        for batch_data in tqdm(ldr_tst):
            # print(batch_data["image"])
            if len(batch_data) == 0:
                continue
            x_batch = batch_data["image"].to(self.device, non_blocking=True)
            outputs = self.net_(x_batch, shap=False)
            
            # convert output from dict-of-list to list of dict, then append
            tmp = {k: outputs[k].tolist() for k in self.tgt_modalities}
            tmp = [{k: tmp[k][i] for k in self.tgt_modalities} for i in range(len(next(iter(tmp.values()))))]
            logits += tmp

        return logits
        
    def predict_proba(self,
        ldr_tst: Any | None = None,
        temperature: float = 1.0,
    ) -> list[dict[str, float]]:
        ''' ... '''
        logits = self.predict_logits(ldr_tst)
        print("got logits")
        return logits, [{k: expit(smp[k] / temperature) for k in self.tgt_modalities} for smp in logits]

    def predict(self,
        ldr_tst: Any | None = None,
    ) -> list[dict[str, int]]:
        ''' ... '''
        logits, proba = self.predict_proba(ldr_tst)
        print("got proba")
        return logits, proba, [{k: int(smp[k] > 0.5) for k in self.tgt_modalities} for smp in proba]