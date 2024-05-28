# coding=utf-8
from audioop import minmax
from torchvision import transforms
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchio as tio
import torch 
import monai

def image_train(dataset, resize_size=256, crop_size=224, augmentation=False):
    if dataset == 'dg5':
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    minmax_normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 0.0005)
    
    if dataset == 'MRI':
        trans_list = []
        if augmentation:
            trans_list.append(transforms.RandomApply(
                            [monai.transforms.RandSpatialCrop(roi_size=(182//4,218//4,182//4), random_center=True, random_size=True),
                            monai.transforms.Resize(spatial_size=(182,218,182))], p=0.5))
        trans_list.append(tio.RandomGamma(p=0.5))
        trans_list.append(tio.RandomBiasField(p=0.25))
        trans_list.append(minmax_normalize)

        return tio.Compose(trans_list)


    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
        transforms.RandomGrayscale(),
        transforms.ToTensor(),
        minmax_normalize
    ])


def image_test(dataset, resize_size=256, crop_size=224):
    if dataset == 'dg5':
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    minmax_normalize = lambda x: (x - x.min()) / (x.max() - x.min() + 0.0005)

    if dataset == 'MRI':
       return minmax_normalize


    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        minmax_normalize
    ])


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count