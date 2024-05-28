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

from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, SmartCacheDataset, load_decathlon_datalist
from monai.transforms import (
    EnsureChannelFirstd,
    AsChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    RandSpatialCropSamplesd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    ToTensord,
)

import torch
import torch.distributed as dist
from torch import nn
import torchio as tio
import random
import functools
from functools import reduce
from PIL import ImageFilter, ImageOps
from monai.utils.type_conversion import convert_to_tensor
from torch.utils.data._utils.collate import default_collate
import sys
sys.path.append('../')
from data import MonaiDataset

# labels = ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']

def collate_handle_corrupted(samples_list, dataset, dtype=torch.half, labels=None):
    if isinstance(samples_list[0], list):
        # samples_list = list(reduce(lambda x,y: x + y, samples_list, []))
        samples_list = [s for s in samples_list if s is not None]
        samples_list = [s for sample in samples_list for s in sample if s is not None]
    print("len(samples_list): ", len(samples_list))
    orig_len = len(samples_list)
    # for the loss to be consistent, we drop samples with NaN values in any of their corresponding crops
    for i, s in enumerate(samples_list):
        print("s is None: ", s is None)
        if s is None:
            continue
        if isinstance(s, dict):
            if 'global_crops' in s.keys() and 'local_crops' in s.keys():
                print(len(s['global_crops']),len(s['local_crops']))
                for c in s['global_crops'] + s['local_crops']:
                    ic(c.size(), torch.isnan(c).any())
                    if torch.isnan(c).any() or c.shape[0] != 1:
                        samples_list[i] = None
                        ic(i, 'removed sample')
                        
            elif 'image' in s.keys():
                for c in s['image']:
                    if torch.isnan(c).any() or c.shape[0] != 1:
                        samples_list[i] = None
                        ic(i, 'removed sample')
        
        elif isinstance(s, torch.Tensor):
            if torch.isnan(s).any() or s.shape[0] != 1:
                samples_list[i] = None
                ic(i, 'removed sample')
                break
    samples_list = list(filter(lambda x: x is not None, samples_list))
    ic(len(samples_list))

    if len(samples_list) == 0:
        # return None
        ic('recursive call')
        return collate_handle_corrupted([dataset[random.randint(0, len(dataset)-1)] for _ in range(orig_len)], dataset, labels=labels)

    if isinstance(samples_list[0], torch.Tensor):
        samples_list = [s for s in samples_list if not torch.isnan(s).any()]
        collated_images = torch.stack([convert_to_tensor(s) for s in samples_list])
        return {"image": collated_images}

    if "image" in samples_list[0]:
        samples_list = [s for s in samples_list if not torch.isnan(s["image"]).any()]
        print('samples list: ', len(samples_list))
        collated_images = torch.stack([convert_to_tensor(s["image"]) for s in samples_list])
        collated_labels = {k: torch.Tensor([s["label"][k] for s in samples_list]) for k in labels}
        return {"image": collated_images,
                "label": collated_labels}
        # return {"image": torch.stack([s["image"] for s in samples_list]).to(dtype)}

    global_crops_list = [crop for s in samples_list for crop in s["global_crops"] if (not torch.isnan(crop).any() and crop.shape[0]==1)]
    local_crops_list = [crop for s in samples_list for crop in s["local_crops"] if (not torch.isnan(crop).any() and crop.shape[0]==1)]

    ic(len(global_crops_list), len(local_crops_list))


    if len(global_crops_list) > 0:
        assert len(set([crop.shape[0] for crop in global_crops_list])) == 1
        collated_global_crops = torch.stack(global_crops_list).to(dtype)
    else:
        collated_global_crops = None
    if len(local_crops_list) > 0:
        assert len(set([crop.shape[0] for crop in local_crops_list])) == 1
        collated_local_crops = torch.stack(local_crops_list).to(dtype)
    else:
        collated_local_crops = None

    # B = len(collated_global_crops)
    # N = n_tokens

    # n_samples_masked = int(B * mask_probability)
    # probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    # upperbound = 0
    # masks_list = []

    # for i in range(0, n_samples_masked):
    #     prob_min = probs[i]
    #     prob_max = probs[i + 1]
    #     masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
    #     upperbound += int(N * prob_max)
    # for i in range(n_samples_masked, B):
    #     masks_list.append(torch.BoolTensor(mask_generator(0)))
    
    # random.shuffle(masks_list)
    # collated_masks = torch.stack(masks_list).flatten(1)
    # mask_indices_list = collated_masks.flatten().nonzero().flatten()

    # masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]
    ic.disable()
    return {
        "collated_global_crops": collated_global_crops,
        "collated_local_crops": collated_local_crops,
        # "collated_masks": collated_masks,
        # "mask_indices_list": mask_indices_list,
        # "masks_weight": masks_weight,
        # "upperbound": upperbound,
        # "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
    }

def monai_collate(samples_list, dataset, dtype=torch.half):
    ic(type(samples_list[0]), len(samples_list))
    orig_len = len(samples_list)
    ic(orig_len, type(samples_list[0]))
    for i, sample in enumerate(samples_list):
        if sample is None:
            continue
        if isinstance(sample[0], dict):
            sample = [s["image"] for s in sample]
        if isinstance(sample[0], list):
            sample = list(reduce(lambda x,y: x + y, sample, []))
        sample_len = len(sample)
        sample = [s for s in sample if not (s is None or torch.isnan(s).any() or s.shape[0] != 1)]
        ic(sample_len, len(sample))
        if len(sample) < sample_len:
            samples_list[i] = None
        else:
            samples_list[i] = sample

    samples_list = [s for s in samples_list if s is not None]
    ic(len(samples_list))
    diff = orig_len - len(samples_list)
    ic(diff)
    if diff > 0:
        ic('recursive call')  
        return monai_collate(samples_list + [dataset[random.randint(0, len(dataset)-1)] for _ in range(diff)], dataset)

    for s in samples_list:
        ic(type(s))
        if isinstance(s, list):
            ic(len(s), [si.size() for si in s])
        elif isinstance(s, torch.Tensor):
            ic(s.size())
    if isinstance(samples_list[0], list):
        samples_list = list(reduce(lambda x,y: x + y, samples_list, []))
    ic(len(samples_list))
    return torch.stack([convert_to_tensor(s) for s in samples_list])

def monai_collate_singles(samples_list, dataset, dtype=torch.half, return_dict=False, labels=None, multilabel=False):
    orig_len = len(samples_list)
    for s in samples_list:
        if isinstance(s, tuple):
            fname, img = s
            if s is None or img is None or img["image"] is None:
                samples_list.remove(s)
        else:
            if s is None or s["image"] is None:
                samples_list.remove(s)

    samples_list = [s for s in samples_list if not torch.isnan(s["image"]).any()]
    diff = orig_len - len(samples_list)
    ic(diff)
    if diff > 0:
        ic('recursive call')  
        return monai_collate_singles(samples_list + [dataset[random.randint(0, len(dataset)-1)] for _ in range(diff)], dataset, return_dict=return_dict, labels=labels, multilabel=multilabel)

    if return_dict:
        collated_dict = {"image": torch.stack([convert_to_tensor(s["image"]) for s in samples_list])}
        if labels:
            if multilabel:
                collated_dict["label"] = {k: torch.Tensor([s["label"][k] for s in samples_list]) for k in labels}
            else:
                collated_dict["label"] = torch.Tensor([s["label"] for s in samples_list])
        return collated_dict
    
    else:
        if isinstance(samples_list[0], tuple):
            # return fnames, imgs
            fnames_list = [s[0] for s in samples_list]
            imgs_list = [convert_to_tensor(s[1]["image"]) for s in samples_list]
            return fnames_list, torch.stack(imgs_list)
        return torch.stack([convert_to_tensor(s["image"]) for s in samples_list])
    

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        backbone.conv3d_transpose = torch.nn.Identity()
        backbone.conv3d_transpose_1 = torch.nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x, is_training=True):
        # ic.enable()
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            ic(start_idx, end_idx)
            ic(torch.stack(x[start_idx: end_idx]).size())
            ic(type(self.backbone))
            _out = self.backbone(torch.stack(x[start_idx: end_idx]), is_training=is_training)
            # _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            ic(type(_out))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            ic(_out.size())
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        ic(output.view(output.shape[0], -1, self.backbone.hidden_size).size())
        # ic.enable()
        return self.head(output.view(output.shape[0], -1, self.backbone.hidden_size))


def minmax_normalize(x):
    eps = torch.finfo(torch.float32).eps if isinstance(x, torch.Tensor) else np.finfo(np.float32).eps
    return (x - x.min()) / (x.max() - x.min() + eps)

def minmax_normalized(x, keys=["image"]):
    for key in keys:
        eps = torch.finfo(torch.float32).eps
        x[key] = torch.nn.functional.relu((x[key] - x[key].min()) / (x[key].max() - x[key].min() + eps))
    return x

class RandomResizedCrop3D(object):
    def __init__(self, size, scale, p):
        self.size = (size,size,size) if isinstance(size, int) else size
        self.Resize3D = tio.transforms.Resize(self.size)
        self.prob = p
        self.scale = scale
    
    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        
        target_scale = random.randrange(**self.scale)
        ic(target_scale)
        target_shape = img.shape * target_scale
        ic(target_shape)

        Crop3D = tio.transforms.CropOrPad(target_shape)
        img = Crop3D(img)
        ic(img.size())
        img = self.Resize3D(img)

        return img


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class Solarization3D(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p, threshold=128):
        self.p = p
        self.threshold = threshold

    def solarize(self, img):
        img[img > self.threshold] *= -1
        img[img < 0] += 255
        return img

    def __call__(self, img):
        if random.random() < self.p:
            return self.solarize(img)
        else:
            return img
        
class Solarization3Dd(Solarization3D):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p, threshold=128, keys=["image"]):
        super().__init__(p, threshold=threshold)
        self.keys = keys

    def solarize(self, img):
        img[img > self.threshold] *= -1
        img[img < 0] += 255
        return img

    def __call__(self, img):
        if random.random() < self.p:
            for k in self.keys:
                img[k] = self.solarize(img[k])
            return img
        else:
            return img

def get_loader(train_list, val_list=None, num_workers=4,
                a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0,
                roi_x=96, roi_y=96, roi_z=96, sw_batch_size=2,
                batch_size=2, distributed=False, cache_dataset=True, smartcache_dataset=False):

    train_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            # Orientationd(keys=["image"], axcodes="RAS"),
            # ScaleIntensityRanged(
            #     keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
            # ),
            minmax_normalized,
            SpatialPadd(keys="image", spatial_size=[roi_x, roi_y, roi_z]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[roi_x, roi_y, roi_z]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[roi_x, roi_y, roi_z],
                num_samples=sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
            # Orientationd(keys=["image"], axcodes="RAS"),
            # ScaleIntensityRanged(
            #     keys=["image"], a_min=a_min, a_max=a_max, b_min=b_min, b_max=b_max, clip=True
            # ),
            minmax_normalized,
            SpatialPadd(keys="image", spatial_size=[roi_x, roi_y, roi_z]),
            CropForegroundd(keys=["image"], source_key="image", k_divisible=[roi_x, roi_y, roi_z]),
            RandSpatialCropSamplesd(
                keys=["image"],
                roi_size=[roi_x, roi_y, roi_z],
                num_samples=sw_batch_size,
                random_center=True,
                random_size=False,
            ),
            ToTensord(keys=["image"]),
        ]
    )

    if cache_dataset:
        print("Using MONAI Cache Dataset")
        train_ds = CacheDataset(data=train_list, transform=train_transforms, cache_rate=0.5, num_workers=num_workers)
    elif smartcache_dataset:
        print("Using MONAI SmartCache Dataset")
        train_ds = SmartCacheDataset(
            data=train_list,
            transform=train_transforms,
            replace_rate=1.0,
            cache_num=2 * batch_size * sw_batch_size,
        )
    else:
        print("Using generic dataset")
        train_ds = MonaiDataset(data=train_list, transform=train_transforms)
    collate_fn = functools.partial(monai_collate, dataset=train_ds)
    val_ds = MonaiDataset(data=val_list, transform=val_transforms)

    if distributed:
        train_sampler = DistributedSampler(dataset=train_ds, even_divisible=True, shuffle=True, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        # val_sampler = DistributedSampler(dataset=val_ds, even_divisible=True, shuffle=False, num_replicas=dist.get_world_size(), rank=dist.get_rank())
    else:
        train_sampler = None
        # val_sampler = None
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler, drop_last=True, collate_fn=collate_fn
    )

    collate_fn = functools.partial(monai_collate, dataset=val_ds)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=True, collate_fn=collate_fn
                            )

    return train_loader, val_loader
    