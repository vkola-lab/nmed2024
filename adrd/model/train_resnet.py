import torch
import numpy as np
import tqdm
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from scipy.special import expit
from copy import deepcopy
from contextlib import suppress
from typing import Any, Self
from icecream import ic

from .. import nn
from ..utils import TransformerTrainingDataset
from ..utils import TransformerValidationDataset
from ..utils import MissingMasker
from ..utils import ConstantImputer
from ..utils import Formatter
from ..utils.misc import ProgressBar
from ..utils.misc import get_metrics_multitask, print_metrics_multitask


class TrainResNet(BaseEstimator):
    ''' ... '''
    def __init__(self,
        src_modalities: dict[str, dict[str, Any]],
        tgt_modalities: dict[str, dict[str, Any]],
        label_fractions: dict[str, float],
        num_epochs: int = 32,
        batch_size: int = 8,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        gamma: float = 0.0,
        criterion: str | None = None,
        device: str = 'cpu',
        cuda_devices: list = [1,2],
        mri_feature: str = 'img_MRI_T1',
        ckpt_path: str = '/home/skowshik/ADRD_repo/adrd_tool/adrd/dev/ckpt/ckpt.pt',
        load_from_ckpt: bool = True,
        save_intermediate_ckpts: bool = False,
        data_parallel: bool = False,
        verbose: int = 0,
    ):  
        ''' ... '''
        # for multiprocessing
        self._rank = 0
        self._lock = None

        # positional parameters
        self.src_modalities = src_modalities
        self.tgt_modalities = tgt_modalities

        # training parameters
        self.label_fractions = label_fractions
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.criterion = criterion
        self.device = device
        self.cuda_devices = cuda_devices
        self.mri_feature = mri_feature
        self.ckpt_path = ckpt_path
        self.load_from_ckpt = load_from_ckpt
        self.save_intermediate_ckpts = save_intermediate_ckpts
        self.data_parallel = data_parallel
        self.verbose = verbose

    def fit(self, x, y):
        ''' ... '''
        # for PyTorch computational efficiency
        torch.set_num_threads(1)

        # set the device for use
        if self.device == 'cuda':
            self.device = "{}:{}".format(self.device, str(self.cuda_devices[0]))

        # initialize model
        if self.load_from_ckpt:
            try:
                print("Loading model from checkpoint...")
                self.load(self.ckpt_path, map_location=self.device)
            except:
                print("Cannot load from checkpoint. Initializing new model...")
                self.load_from_ckpt = False

        # initialize model
        if not self.load_from_ckpt:
            self.net_ = nn.ResNetModel(
                self.tgt_modalities,
                mri_feature = self.mri_feature
            )
            # intialize model parameters using xavier_uniform
            for p in self.net_.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)

        self.net_.to(self.device)

        # Initialize the number of GPUs
        if self.data_parallel and torch.cuda.device_count() > 1:
            print("Available", torch.cuda.device_count(), "GPUs!")
            self.net_ = torch.nn.DataParallel(self.net_, device_ids=self.cuda_devices)


        # split dataset
        x_trn, x_vld, y_trn, y_vld = train_test_split(
            x, y, test_size = 0.2, random_state = 0,
        )

        # initialize dataset and dataloader
        dat_trn = TransformerTrainingDataset(
            x_trn, y_trn,
            self.src_modalities,
            self.tgt_modalities,
            dropout_rate = .5,
            dropout_strategy = 'compensated',
            mri_feature = self.mri_feature,
        )

        dat_vld = TransformerValidationDataset(
            x_vld, y_vld,
            self.src_modalities,
            self.tgt_modalities,
            mri_feature = self.mri_feature,
        )

        # ic(dat_trn[0])

        ldr_trn = torch.utils.data.DataLoader(
            dat_trn,
            batch_size = self.batch_size,
            shuffle = True,
            drop_last = False,
            num_workers = 0,
            collate_fn = TransformerTrainingDataset.collate_fn,
            # pin_memory = True
        )

        ldr_vld = torch.utils.data.DataLoader(
            dat_vld,
            batch_size = self.batch_size,
            shuffle = False,
            drop_last = False,
            num_workers = 0,
            collate_fn = TransformerTrainingDataset.collate_fn,
            # pin_memory = True
        )

        # initialize optimizer
        optimizer = torch.optim.AdamW(
            self.net_.parameters(),
            lr = self.lr,
            betas = (0.9, 0.98),
            weight_decay = self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64, verbose=(self.verbose > 2))
        
        # initialize loss function (binary cross entropy)
        loss_fn = {}

        for k in self.tgt_modalities:
            alpha = pow((1 - self.label_fractions[k]), self.gamma)
            # if alpha < 0.5:
            #     alpha = -1
            loss_fn[k] = nn.SigmoidFocalLoss(
                alpha = alpha,
                gamma = self.gamma,
                reduction = 'none'
            )

        # to record the best validation performance criterion
        if self.criterion is not None:
            best_crit = None

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

        # Define a hook function to print and store the gradient of a layer
        def print_and_store_grad(grad, grad_list):
            grad_list.append(grad)
            # print(grad)

        # grad_list = []
        # self.net_.module.img_net_.featurizer.down_tr64.ops[0].conv1.weight.register_hook(lambda grad: print_and_store_grad(grad, grad_list))
        # self.net_.module.modules_emb_src['gender'].weight.register_hook(lambda grad: print_and_store_grad(grad, grad_list))


        # training loop
        for epoch in range(self.num_epochs):
            # progress bar for batch loops
            if self.verbose > 1: 
                pbr_batch = ProgressBar(len(dat_trn), 'Epoch {:03d} (TRN)'.format(epoch))

            # set model to train mode
            torch.set_grad_enabled(True)
            self.net_.train()

            scores_trn, y_true_trn = [], []
            losses_trn = [[] for _ in self.tgt_modalities]
            for x_batch, y_batch, mask in ldr_trn:

                # mount data to the proper device
                x_batch = {k: x_batch[k].to(self.device) for k in x_batch}
                y_batch = {k: y_batch[k].to(torch.float).to(self.device) for k in y_batch}

                # forward
                outputs = self.net_(x_batch)

                # calculate multitask loss
                loss = 0
                for i, k in enumerate(self.tgt_modalities):
                    loss_task = loss_fn[k](outputs[k], y_batch[k])
                    loss += loss_task.mean()
                    losses_trn[i] += loss_task.detach().cpu().numpy().tolist()

                # backward
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                ''' TODO: change array to dictionary later '''
                outputs = torch.stack(list(outputs.values()), dim=1)
                y_batch = torch.stack(list(y_batch.values()), dim=1)

                # save outputs to evaluate performance later
                scores_trn.append(outputs.detach().to(torch.float).cpu())
                y_true_trn.append(y_batch.cpu())

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

            # set scheduler
            scheduler.step()

            # calculate and print training performance metrics
            scores_trn = torch.cat(scores_trn)
            y_true_trn = torch.cat(y_true_trn)
            y_pred_trn = (scores_trn > 0).to(torch.int)
            y_prob_trn = torch.sigmoid(scores_trn)
            met_trn = get_metrics_multitask(
                y_true_trn.numpy(),
                y_pred_trn.numpy(),
                y_prob_trn.numpy()
            )

            # add loss to metrics
            for i in range(len(self.tgt_modalities)):
                met_trn[i]['Loss'] = np.mean(losses_trn[i])

            if self.verbose > 2:
                print_metrics_multitask(met_trn)

            # progress bar for validation
            if self.verbose > 1:
                pbr_batch = ProgressBar(len(dat_vld), 'Epoch {:03d} (VLD)'.format(epoch))

            # set model to validation mode
            torch.set_grad_enabled(False)
            self.net_.eval()

            scores_vld, y_true_vld = [], []
            losses_vld = [[] for _ in self.tgt_modalities]
            for x_batch, y_batch, mask in ldr_vld:
                # mount data to the proper device
                x_batch = {k: x_batch[k].to(self.device) for k in x_batch}
                y_batch = {k: y_batch[k].to(torch.float).to(self.device) for k in y_batch}

                # forward
                outputs = self.net_(x_batch)

                # calculate multitask loss
                for i, k in enumerate(self.tgt_modalities):
                    loss_task = loss_fn[k](outputs[k], y_batch[k])
                    losses_vld[i] += loss_task.detach().cpu().numpy().tolist()

                ''' TODO: change array to dictionary later '''
                outputs = torch.stack(list(outputs.values()), dim=1)
                y_batch = torch.stack(list(y_batch.values()), dim=1)

                # save outputs to evaluate performance later
                scores_vld.append(outputs.detach().to(torch.float).cpu())
                y_true_vld.append(y_batch.cpu())

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
            y_pred_vld = (scores_vld > 0).to(torch.int)
            y_prob_vld = torch.sigmoid(scores_vld)
            met_vld = get_metrics_multitask(
                y_true_vld.numpy(),
                y_pred_vld.numpy(),
                y_prob_vld.numpy()
            )

            # add loss to metrics
            for i in range(len(self.tgt_modalities)):
                met_vld[i]['Loss'] = np.mean(losses_vld[i])

            if self.verbose > 2:
                print_metrics_multitask(met_vld)

            # save the model if it has the best validation performance criterion by far
            if self.criterion is None: continue
            
            # is current criterion better than previous best?
            curr_crit = np.mean([met_vld[i][self.criterion] for i in range(len(self.tgt_modalities))])
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
                if self.save_intermediate_ckpts:
                    print("Saving the model...")
                    self.save(self.ckpt_path)

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
    ) -> list[dict[str, float]]:
        '''
        The input x can be a single sample or a list of samples.
        '''
        # input validation
        check_is_fitted(self)
        
        # for PyTorch computational efficiency
        torch.set_num_threads(1)

        # set model to eval mode
        torch.set_grad_enabled(False)
        self.net_.eval()

        # number of samples to evaluate
        n_samples = len(x)

        # format x
        fmt = Formatter(self.src_modalities)
        x = [fmt(smp) for smp in x]

        # generate missing mask (BEFORE IMPUTATION)
        msk = MissingMasker(self.src_modalities)
        mask = [msk(smp) for smp in x]

        # reformat x and then impute by 0s
        imp = ConstantImputer(self.src_modalities)
        x = [imp(smp) for smp in x]

        # convert list-of-dict to dict-of-list
        x = {k: [smp[k] for smp in x] for k in self.src_modalities}
        mask = {k: [smp[k] for smp in mask] for k in self.src_modalities}
        
        # to tensor
        x = {k: torch.as_tensor(np.array(v)).to(self.device) for k, v in x.items()}
        mask = {k: torch.as_tensor(np.array(v)).to(self.device) for k, v in mask.items()}

        # calculate logits
        logits = self.net_(x)

        # convert dict-of-list to list-of-dict
        logits = {k: logits[k].tolist() for k in self.tgt_modalities}
        logits = [{k: logits[k][i] for k in self.tgt_modalities} for i in range(n_samples)]

        return logits
        
    def predict_proba(self,
        x: list[dict[str, Any]],
        temperature: float = 1.0
    ) -> list[dict[str, float]]:
        ''' ... '''
        # calculate logits
        logits = self.predict_logits(x)
        
        # convert logits to probabilities and 
        proba = [{k: expit(smp[k] / temperature) for k in self.tgt_modalities} for smp in logits]
        return proba

    def predict(self,
        x: list[dict[str, Any]],
    ) -> list[dict[str, int]]:
        ''' ... '''
        proba = self.predict_proba(x)
        return [{k: int(smp[k] > 0.5) for k in self.tgt_modalities} for smp in proba]

    def save(self, filepath: str) -> None:
        ''' ... '''
        check_is_fitted(self)
        if self.data_parallel:
            state_dict = self.net_.module.state_dict()
        else:
            state_dict = self.net_.state_dict()

        # attach model hyper parameters
        state_dict['src_modalities'] = self.src_modalities
        state_dict['tgt_modalities'] = self.tgt_modalities
        state_dict['mri_feature'] = self.mri_feature

        torch.save(state_dict, filepath)

    def load(self, filepath: str, map_location: str='cpu') -> None:
        ''' ... '''
        # load state_dict
        state_dict = torch.load(filepath, map_location=map_location)

        # load data modalities
        self.src_modalities = state_dict.pop('src_modalities')
        self.tgt_modalities = state_dict.pop('tgt_modalities')

        # initialize model
        self.net_ = nn.ResNetModel(
                self.tgt_modalities,
                mri_feature = state_dict.pop('mri_feature')
            )

        # load model parameters
        self.net_.load_state_dict(state_dict)
        self.net_.to(self.device) 

    @classmethod
    def from_ckpt(cls, filepath: str, device='cpu') -> Self:
        ''' ... '''
        obj = cls(None, None, None,device=device)
        obj.load(filepath)
        return obj