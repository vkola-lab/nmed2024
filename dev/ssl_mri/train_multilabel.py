# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import argparse
import datetime
import numpy as np
from functools import partial

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torchvision import transforms
from timm.utils import AverageMeter

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler
import utils
from utils.data_utils import monai_collate_singles, minmax_normalized
from utils.dist_utils import reduce_tensor, save_on_master, init_distributed_mode, has_batchnorms, get_world_size
from utils.ops import clip_gradients, cancel_gradients_last_layer
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import icecream
from icecream import ic, install
import sys
sys.path.append('/projectnb/ivc-ml/dlteif/adrd_tool/')
from adrd.nn.focal_loss import AsymmetricLoss
from monai.networks.nets import SwinUNETR
# from monai.inferers import sliding_window_inferer

install()
ic.configureOutput(includeContext=True)
ic.disable()

import monai
from monai.transforms import (
    LoadImaged,
    Compose,
    CropForegroundd,
    Resized,
    EnsureChannelFirstd,
    Spacingd,
    CenterSpatialCropd,
)

from monai.networks.nets import ViTAutoEnc
from monai.data import (
    CacheDataset,
    load_decathlon_datalist,
)

from data.mri_dataset import read_csv, read_df
import functools
from adrd.utils.misc import print_metrics_multitask, get_metrics_multitask
from tqdm import tqdm

def parse_option():
    parser = argparse.ArgumentParser("ViT Self-Supervised Learning", add_help=False)

    # Set Paths for running SSL training
    parser.add_argument('--train_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--val_path', default='/path/to/imagenet/val/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--test_path', default='/path/to/imagenet/test/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument("--num_labels", default=3, type=int, help="Number of labels")
    parser.add_argument('--dataset', type=str,
        help='Please specify dataset name')
    parser.add_argument("--logdir_path", default="/to/be/defined", type=str, help="output log directory")
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['SwinUNETR', 'ViTAutoEnc', 'Feature3D_ViT2D_V2', 'vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'],
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--resume', action='store_true', help="Set to True to resume training from latest checkpoint.")
    parser.add_argument('--checkpoint_path', default=None, type=str, help="Path to checkpoint.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')

    # Distributed Training
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--gpu', type=int, default=None, help="GPU id")
    parser.add_argument('--dropout_rate', type=float, default=0.1, help="stochastic depth rate")


    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # DL Training Hyper-parameters
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=1, type=int, help="batch size for single GPU")
    parser.add_argument("--patch_size", default=16, type=int, help="patch size for ViT")
    parser.add_argument("--num_heads", default=12, type=int, help="number of heads in the ViT.")
    parser.add_argument("--image_size", default=96, type=int, help="image size")
    parser.add_argument("--base_lr", default=5e-4, type=float, help="base learning rate")
    parser.add_argument('--use_fp16', type=bool, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument("--seed", default=19, type=int, help="seed")
    parser.add_argument("--deterministic", help="set seed for deterministic training", action="store_true")
    
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')

    args = parser.parse_args()
    return args


def main(args):
    # Define training transforms
    flip_and_jitter = monai.transforms.Compose([
            monai.transforms.RandAxisFlipd(keys=["image"], prob=0.5),
            transforms.RandomApply(
                [
                   monai.transforms.RandAdjustContrastd(keys=["image"], gamma=(-0.3,0.3)), # Random Gamma => randomly change contrast by raising the values to the power log_gamma 
                   monai.transforms.RandBiasFieldd(keys=["image"]), # Random Bias Field artifact
                   monai.transforms.RandGaussianNoised(keys=["image"]),

                ],
                p=1.0
            ),
        ])

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            CropForegroundd(keys=["image"], source_key="image"),
            monai.transforms.RandScaleCropd(keys=["image"], roi_scale=0.4, max_roi_scale=1, random_size=True, random_center=True),
            monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=args.image_size),
            flip_and_jitter,
            monai.transforms.RandGaussianSmoothd(keys=["image"], prob=0.5),
            minmax_normalized,
        ]            
    )

    # train_transforms = DataAugmentationDINO3D(
    #         args.global_crops_scale,
    #         args.local_crops_scale,
    #         args.local_crops_number,
    #         args.image_size
    #         )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            CropForegroundd(keys=["image"], source_key="image"),
            # CenterSpatialCropd(keys=["image"], roi_size=(args.image_size,)*3),
            Resized(keys=["image"], spatial_size=(args.image_size*2,)*3),
            monai.transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=args.image_size),
            minmax_normalized,
        ]
    )

    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.swinunetr = model

        def forward(self, x_in):
            hidden_states_out = self.swinunetr.swinViT(x_in, self.swinunetr.normalize)
            ic(h.size() for h in hidden_states_out)
            enc0 = self.swinunetr.encoder1(x_in)
            enc1 = self.swinunetr.encoder2(hidden_states_out[0])
            enc2 = self.swinunetr.encoder3(hidden_states_out[1])
            enc3 = self.swinunetr.encoder4(hidden_states_out[2])
            dec4 = self.swinunetr.encoder10(hidden_states_out[4])
            print(enc0.size(), enc1.size(), enc2.size(), enc3.size(), dec4.size())
            
            return dec4

    # Define the model, losses and optimizer
    if args.arch == 'ViTAutoEnc':
        model = ViTAutoEnc(
            in_channels=1,
            img_size=(args.image_size,)*3,
            patch_size=(args.patch_size,)*3,
            pos_embed="conv",
            num_heads=args.num_heads,
            hidden_size=768,
            mlp_dim=3072,
            dropout_rate=args.dropout_rate,
        )

        model.conv3d_transpose = torch.nn.Identity()
        model.conv3d_transpose_1 = torch.nn.Identity()
    
    elif args.arch == 'SwinUNETR':
        model = SwinUNETR(
                in_channels=1,
                out_channels=1,
                img_size=(args.image_size,)*3,
                feature_size=48,
                use_checkpoint=True,
            )
        # model.decoder5 = torch.nn.Identity()
        # model.decoder4 = torch.nn.Identity()
        # model.decoder3 = torch.nn.Identity()
        # model.decoder2 = torch.nn.Identity()
        # model.decoder1 = torch.nn.Identity()


        pretrained_pth = "/projectnb/ivc-ml/dlteif/pretrained_models/model_swinvit.pt"
        model_dict = torch.load(pretrained_pth, map_location="cpu")
        model_dict["state_dict"] = {k.replace("swinViT.", "module."): v for k, v in model_dict["state_dict"].items()}
        ic(model_dict["state_dict"].keys())
        model.load_from(model_dict)

        model = ModelWrapper(model)

        # class ModelWrapper(torch.nn.Module):
        #     def __init__(self, model, *args, **kwargs) -> None:
        #         super().__init__(*args, **kwargs)
        #         self.model = model

        #     def forward(self, x):
        #         return self.model(x, output_only=True)
    print(type(model))
    print("Number of params: ", sum(p.numel() for p in model.parameters()))

    # labels = ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']
    labels = ['NC', 'MCI', 'DE']
    classifier = ConvClassifier(768, depth=3, num_labels=len(labels), kernel_size=1)
    # freeze backbone
    # for name, module in model.named_modules():
    #     ic(name)
    #     ic(type(module))
    
    
    if args.resume:
        ckpt_path = args.checkpoint_path if args.checkpoint_path is not None else os.path.join(args.output, f'checkpoint.pth')
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if 'dino_loss' in state_dict:
            print(state_dict['dino_loss'])
        print('epoch: ', state_dict['epoch'])
        start_epoch = state_dict['epoch'] + 1
        if args.checkpoint_key is not None and args.checkpoint_key in ['teacher', 'student'] and args.checkpoint_key in state_dict:
            print(f"Take key {args.checkpoint_key} in provided checkpoint dict")
            state_dict = state_dict[args.checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found and loaded with msg: {}'.format(msg))
        else:
            model.load_state_dict(state_dict['backbone'], strict=False)
            classifier.load_state_dict(state_dict['classifier'])
        
        print('Resume training from epoch {}'.format(start_epoch))
    
    if has_batchnorms(model):
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)    
    device = torch.device(f"cuda:{args.gpu}" if args.gpu else "cpu")
    model.cuda()
    
    for p in model.parameters():
        p.requires_grad = False

    if has_batchnorms(model):
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # model.cuda()
    classifier = classifier.cuda()
    print('model sent to cuda')
    if has_batchnorms(classifier):
        torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)

    optimizer = torch.optim.Adam(list(classifier.parameters()),
                                 lr=args.base_lr * (args.batch_size * get_world_size()) / 256.,
                                 )
    
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()
    
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
    # )
    # model_without_ddp = model.module
    classifier = torch.nn.parallel.DistributedDataParallel(
        classifier, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True
    )
    classifier_without_ddp = classifier.module
    print('model is ready..')

    n_parameters = sum(p.numel() for p in torch.nn.Sequential(model, classifier).parameters() if p.requires_grad)
    print("number of params: {}".format(n_parameters))


    # Build the data loader
    train_list = read_df(args.train_path, return_dicts=True, labels=labels, multilabel=True)
    val_list = read_df(args.val_path, return_dicts=True, labels=labels, multilabel=True)
    test_list = read_df(args.test_path, return_dicts=True, labels=labels, multilabel=True)
    print(len(train_list), len(val_list), len(test_list))
    train_labels = [item['label'] for item in train_list]
    print(len(train_labels))
    counts = [sum([float(l[label]) for l in train_labels]) for label in labels]
    print(counts)
    weights = torch.Tensor([len(train_labels) / counts[i] if counts[i] > 0 else 0 for i in range(len(counts))])
    print(weights)
    # loss_funcs = [torch.nn.BCELoss(weight=weights.cuda(non_blocking=True))]
    # loss_funcs = [AsymmetricLoss()]
    # initialize asymmetric focal loss
    loss_fn = {}
    for k in labels:
        # alpha = pow((1 - self.label_fractions[k]), self.gamma)
        alpha = 0.5
        if k == 'NC'  or k == 'DE' or k == 'AD' : 
            loss_fn[k] = AsymmetricLoss(gamma_neg=2, gamma_pos=2, alpha=alpha)
        else:
            if k == 'ODE' or k=='TBI' or k=='SEF' or k=='NPH':
                loss_fn[k] = AsymmetricLoss(gamma_neg=5, gamma_pos=1, alpha=alpha)
            else:
                loss_fn[k] = AsymmetricLoss(gamma_neg=3, gamma_pos=1, alpha=alpha)
    lambda_coeff = 0.00001
    margin_loss = torch.nn.MarginRankingLoss(reduction='sum')

    print("Total training data are {} and validation data are {}".format(len(train_list),len(val_list)))

    train_dataset = monai.data.Dataset(data=train_list, transform=train_transforms)
    val_dataset = monai.data.Dataset(data=val_list, transform=val_transforms)
    val_dataset = monai.data.Dataset(data=test_list, transform=val_transforms)

    train_sampler = DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True
    )

    val_sampler = DistributedSampler(
        val_dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False
    )
    collate_fn = functools.partial(monai_collate_singles, dataset=train_dataset, dtype=torch.FloatTensor, return_dict=True, labels=labels, multilabel=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=functools.partial(monai_collate_singles, dataset=val_dataset, dtype=torch.FloatTensor, return_dict=True, labels=labels, multilabel=True),
    )
    
    
    # Data Loaders Built
    
    if args.evaluate:
        val_loss_avg = validate(data_loader=val_loader, 
                                model=model, 
                                classifier=classifier,
                                epoch=0, 
                                fp16_scaler=fp16_scaler, 
                                loss_functions=loss_fn, 
                                labels_list=labels)
        return


    print("Training Begins ...")

    start_time = time.time()
    val_loss_best = 1e9

    for epoch in range(args.epochs):
        train_loss_avg = train_one_epoch(
            args=args,
            model=model,
            classifier=classifier,
            data_loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            fp16_scaler=fp16_scaler,
            loss_functions=loss_fn,
            labels_list=labels,
            margin_loss=margin_loss,
            lambda_coeff=lambda_coeff
        )

        val_loss_avg = validate(data_loader=val_loader, 
                                model=model, 
                                classifier=classifier,
                                epoch=epoch, 
                                fp16_scaler=fp16_scaler, 
                                loss_functions=loss_fn, 
                                labels_list=labels)

        if dist.get_rank() == 0:
            log_writer.add_scalar("Validation/loss_total", scalar_value=val_loss_avg, global_step=epoch)
            log_writer.add_scalar("Train/loss_total", scalar_value=train_loss_avg, global_step=epoch)
            
        save_dict = {
            'backbone': model.state_dict(),
            'classifier': classifier_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'train_loss': train_loss_avg,
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        save_on_master(save_dict, os.path.join(args.output, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            save_on_master(save_dict, os.path.join(args.output, f'checkpoint{epoch:04}.pth'))
        if val_loss_avg <= val_loss_best:
            save_on_master(save_dict, os.path.join(args.output, f'checkpoint_best.pth'))


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Total time: {datetime.timedelta(seconds=int(total_time))}")


def train_one_epoch(args, model, classifier, data_loader, optimizer, epoch, fp16_scaler, loss_functions, labels_list, lambda_coeff=None, margin_loss=None):
    model.train()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()

    start = time.time()
    end = time.time()
    
    scores_trn, y_true_trn, y_mask_trn = [], [], []
    losses_trn = [[] for _ in loss_functions.keys()]
    for idx, batch_data in enumerate(tqdm(data_loader)):
        
        # inputs, inputs_2, gt_input = (
        #     batch_data["image"].cuda(non_blocking=True),
        #     batch_data["image_2"].cuda(non_blocking=True),
        #     batch_data["gt_image"].cuda(non_blocking=True),
        # )
        # print(batch_data.keys())
        inputs = batch_data["image"].cuda(non_blocking=True)
        print(inputs.size())

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            feats = model(inputs)
            if isinstance(feats, tuple):
                feats = feats[0]
            print('feats: ', feats.size(), torch.isnan(feats).any())
            outputs = classifier(feats)
            print('outputs: ', outputs.size(), torch.isnan(outputs).any())

            print(outputs.size(), len(batch_data["label"]))
            labels = {k: v.cuda(non_blocking=True) for k,v in batch_data["label"].items()}
            # y_mask = {k: 1.0 for k in y_mask}
            
            print(outputs.device)

            loss = 0
            if epoch < 10:
                loss = 0
            else:
                for i, k in enumerate(loss_functions.keys()):
                    if i>2 :
                        for ii, kk in enumerate(loss_functions.keys()):
                            if ii>i:
                                loss += lambda_coeff*margin_loss(torch.sigmoid(outputs[:,i]),torch.sigmoid(outputs[:,ii]),labels[k]-labels[kk])

            for i, k in enumerate(loss_functions.keys()):
                loss_task = loss_functions[k](outputs[:,i], labels[k].cuda(non_blocking=True), epoch=epoch) * 10
                loss += loss_task.mean()
                losses_trn[i] += loss_task.detach().cpu().numpy().tolist()

            
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(model, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                # unscale the gradients of optimizer's assigned params in-place
                param_norms = clip_gradients(model, args.clip_grad)
            cancel_gradients_last_layer(epoch, model,
                                            args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        
        
        torch.cuda.synchronize()
        total_loss_t = reduce_tensor(loss)

        loss_meter.update(total_loss_t.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        lr = optimizer.param_groups[0]["lr"]
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        etas = batch_time.avg * (num_steps - idx)
        print(
            f"Train: [{epoch}/{args.epochs}][{idx}/{num_steps}]\t"
            f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
            f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
            f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
            f"mem {memory_used:.0f}MB"
        )

        ''' TODO: change array to dictionary later '''
        y_batch = torch.stack(list(labels.values()), dim=1)
        y_mask = torch.ones_like(y_batch)


        # save outputs to evaluate performance later
        scores_trn.append(outputs.detach().to(torch.float).cpu())
        y_true_trn.append(y_batch.cpu())
        y_mask_trn.append(y_mask.cpu())

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
        y_mask_trn.numpy(),
    )

    # add loss to metrics
    for i in range(len(labels_list)):
        met_trn[i]['Loss'] = np.mean(losses_trn[i])
            
    print({f"Train loss {list(labels)[i]}": met_trn[i]['Loss']  for i in range(len(labels_list))})
    print({f"Train Balanced Accuracy {list(labels)[i]}": met_trn[i]['Balanced Accuracy']  for i in range(len(labels_list))})

    print_metrics_multitask(met_trn)

    epoch_time = time.time() - start
    print(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")

    return loss_meter.avg


@torch.no_grad()
def validate(data_loader, model, classifier, epoch, fp16_scaler, loss_functions, labels_list):
    model.eval()
    loss_meter = AverageMeter()
    
    scores_vld, y_true_vld, y_mask_vld = [], [], []
    losses_vld = [[] for _ in labels_list]
    for idx, batch_data in enumerate(tqdm(data_loader)):

        with torch.no_grad():    
            inputs = batch_data["image"].cuda(non_blocking=True)
            print('inputs: ', inputs.size(), torch.isnan(inputs).any())
            feats = model(inputs)
            if isinstance(feats, tuple):
                feats = feats[0]
            print('feats: ', feats.size(), torch.isnan(feats).any())
            outputs = classifier(feats)
            print('outputs: ', outputs.size(), torch.isnan(outputs).any())

            labels = {k: v.cuda(non_blocking=True) for k,v in batch_data["label"].items()}
            
            val_loss = 0
            for i,k in enumerate(loss_functions.keys()):
                loss_task = loss_functions[k](outputs[:,i], labels[k].cuda(non_blocking=True), epoch=idx) * 10
                # print(f'{k} task loss: {loss_task}')
                val_loss += loss_task.mean()
                losses_vld[i] += loss_task.detach().cpu().numpy().tolist()

            loss = reduce_tensor(val_loss).mean()
            loss_meter.update(loss.item(), inputs.size(0))
        
        print(f"Test: [{idx}/{len(data_loader)}]\t" f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t")


        y_batch = torch.stack(list(labels.values()), dim=1)
        y_mask = torch.ones_like(y_batch)

        # save outputs to evaluate performance later
        scores_vld.append(outputs.detach().to(torch.float).cpu())
        y_true_vld.append(y_batch.cpu())
        y_mask_vld.append(y_mask.cpu())

        # if fp16_scaler is None:
        #     if args.clip_grad:
        #         param_norms = utils.clip_gradients(model, args.clip_grad)
        # else:
        #     fp16_scaler.scale(loss).backward()
        #     if args.clip_grad:
        #         # unscale the gradients of optimizer's assigned params in-place
        #         param_norms = utils.clip_gradients(model, args.clip_grad)
        #     utils.cancel_gradients_last_layer(epoch, model,
        #                                     args.freeze_last_layer)
        #     fp16_scaler.update()


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
        y_mask_vld.numpy(),
    )

    # add loss to metrics
    for i in range(len(labels_list)):
        met_vld[i]['Loss'] = np.mean(losses_vld[i])
        
    print({f"Validation loss {list(labels_list)[i]}": met_vld[i]['Loss'] for i in range(len(labels_list))})
    print({f"Validation Balanced Accuracy {list(labels_list)[i]}": met_vld[i]['Balanced Accuracy']  for i in range(len(labels_list))})


    print(f" * Val Loss {loss_meter.avg:.3f}")

    print_metrics_multitask(met_vld)

    return loss_meter.avg

class LinearClassifier(torch.nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000, depth=1):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        layers = []
        for i in range(depth):
            dim_out = dim // 2
            layers.append(
                torch.nn.Linear(dim, dim_out)
            )
        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(dim, num_labels)
        ])
        self.linear = torch.nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


class ConvClassifier(torch.nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000, depth=1, kernel_size=1):
        super(ConvClassifier, self).__init__()
        self.num_labels = num_labels
        self.layers = torch.nn.ModuleList()
        for i in range(depth):
            if i == depth - 1:
                dim_out = num_labels
                ks = 3
            else:
                dim_out = dim // 2
                ks = 1

            self.layers.append(
                torch.nn.Conv3d(in_channels=dim, out_channels=dim_out, kernel_size=ks)
            )
            dim = dim_out

        self.sigm = torch.nn.Sigmoid()

        for l in self.layers:
            torch.nn.init.xavier_uniform(l.weight)
            l.bias.data.fill_(0.01)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            print(x.size())
        B = x.size(0)
        return self.sigm(x.view(B,-1))

if __name__ == "__main__":
    args = parse_option()
    init_distributed_mode(args)

    seed = args.seed + dist.get_rank()

    if args.deterministic:
        torch.manual_seed(seed)
        torch.manual_seed_all(seed)
        np.random.seed(seed)

    cudnn.benchmark = True

    if dist.get_rank() == 0:
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
        if not os.path.exists(args.logdir_path):
            os.makedirs(args.logdir_path, exist_ok=True)

    if dist.get_rank() == 0:
        log_writer = SummaryWriter(log_dir=args.logdir_path)

    main(args)