# coding=utf-8
import os
import numpy as np
import torch
import sys
import random
# sys.path.append('..')
# from datautil.util import Nmax
from . import util as imgutil
from .util import rgb_loader, l_loader
# import util as imgutil
# from util import rgb_loader, l_loader

from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader
import pandas as pd
from torch.utils.data import Dataset

# from mmap_ninja.ragged import RaggedMmap
from icecream import ic

# import util as imgutil
# from util import rgb_loader, l_loader
def Nmax(test_envs, d):
    for i in range(len(test_envs)):
        if d < test_envs[i]:
            return i
    return len(test_envs)

def read_csv(df, feature, num_classes=3):
    print('------------read csv-------------')
    print('feature: ', feature)
    # print('dataset: ', dataset)
    
    # df = pd.read_csv(filename)

    # filenames = [''.join([row['path'],row['filename']]).replace('.nii', '.npy') for _,row in df.iterrows()]
    filenames = []
    for _, row in df.iterrows():
        print(row[feature])
        if isinstance(row[feature], list):
            filenames.extend(row[feature])
        # else:
        #     filenames.append(None)

    if num_classes == 3:
        print('getting labels for 3way classification')
        # labels = torch.LongTensor([0 if int(row['NC']) else (1 if int(row['MCI']) else 2) for _,row in df.iterrows()])
        labels = []
        for _,row in df.iterrows():
            if isinstance(row['filename'], str) and row['filename'].endswith('.npy'):
                if np.isnan(row['NC']) and int(row['NC']):
                    labels.append(0)
                elif np.isnan(row['MCI']) and int(row['MCI']):
                    labels.append(1)
                else:
                    labels.append(2)
            else:
                labels.append(-1)
        labels = torch.LongTensor(labels)

    else:
        print('getting labels for 2way classification')
        labels = torch.LongTensor([0 if int(row['NC']) else 2 for _,row in df.iterrows()])
    print('-> uniques: ', torch.unique(labels,sorted=True))

    return filenames, labels

DEFAULT_INPUT_FILE_NAME = 'input.data'
DEFAULT_LABELS_FILE_NAME = 'labels.data'

class ImageDataset(Dataset):
    def __init__(self, feature, task, root_dir, domain_name, domain_label=-1, labels=None, transform=None, target_transform=None, indices=None, test_envs=[], mode='Default', df=None, num_classes=3, mmap=False, stripped=False):
        self.domain_num = 0
        self.task = task
        self.domain_name = domain_name
        self.df = df
        if self.task == 'img_dg':
            self.imgs = ImageFolder(root_dir+domain_name).imgs
            self.fnames = [item[0] for item in self.imgs]
            self.labels = [item[1] for item in self.imgs]
        elif self.task == 'mri_dg':
            # print('filename: ', filename)
            self.fnames, self.labels = read_csv(df, feature, num_classes=num_classes)   
            print('fnames: ', len(self.fnames)) 
            print('Not None: ', len([f for f in self.fnames if f is not None]))
        self.mmap = mmap
        # self.mmap_input_path = os.path.join(root_dir, DEFAULT_INPUT_FILE_NAME)
        # self.mmap_labels_path = os.path.join(root_dir, DEFAULT_LABELS_FILE_NAME)
        if self.mmap:
            self.x = np.asarray([np.memmap(f.replace('.nii', '.npy'), dtype=np.float32, mode='r+', shape=(182,218,182)) for f in self.fnames])
        else:
            self.x = self.fnames if not stripped else [f.replace('.nii', '_stripped.nii') for f in self.fnames]
        
        # self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        if indices is None:
            self.indices = np.arange(len(self.fnames))
        else:
            self.indices = indices
        if mode == 'Default':
            self.loader = default_loader
        elif mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.dlabels = np.ones(self.labels.shape) * \
            (domain_label-Nmax(test_envs, domain_label))

    def set_labels(self, tlabels=None, label_type='domain_label'):
        assert len(tlabels) == len(self.x)
        if label_type == 'domain_label':
            self.dlabels = tlabels
        elif label_type == 'class_label':
            self.labels = tlabels

    def target_trans(self, y):
        if self.target_transform is not None:
            return self.target_transform(y)
        else:
            return y

    def input_trans(self, x):
        if self.transform is not None:
            return self.transform(x)
        else:
            return x

    def __getitem__(self, index):
        index = self.indices[index]
        try:
            if self.task == 'mri_dg':
                # print(len(self.x), index)
                # if self.mmap:
                #     data = self.x[index]
                # else:
                #     data = np.load(self.x[index].replace('.nii', '.npy'))
                #     # data = np.memmap(self.x[index].replace('.nii', '.npy'), dtype=np.float32, mode='r+', shape=(182,218,182))
                # # ic(data.shape, data.min(), data.max())
                # data = np.expand_dims(data, axis=0)
                # # print('->', data.shape)
                # img = self.input_trans(torch.from_numpy(data))
                # img = self.input_trans(data)
                img = self.input_trans({"image": self.x[index]})
                return self.fnames[index].replace('.nii', '.npy'), img["image"]
            else:
                img = self.input_trans(self.loader(self.x[index]))
            ctarget = self.target_trans(self.labels[index])
            dtarget = self.target_trans(self.dlabels[index])
            return self.fnames[index].replace('.nii', '.npy'), img, ctarget, dtarget
        except:
            return self.fnames[index].replace('.nii', '.npy'), None

    def __len__(self):
        return len(self.indices)

    def get_sample_weights(self):
        print('------------def get_sample_weights--------------')
        label_list = self.labels.tolist()
        count = float(len(label_list))
        print('total count: ', count)
        print(sorted(list(set(label_list))))
            
        uniques = sorted(list(set(label_list)))
        print('uniques: ',  uniques)
        counts = [float(label_list.count(i)) for i in uniques]
        print('counts: ', counts)
        
        weights = [count / counts[i] for i in label_list]
        # print('weights: ', weights)
        return weights, counts

    def _init_mmap(self, path, dtype, shape, remove_existing=False):
        open_mode = 'r+'
        if remove_existing:
            open_mode='w+'

        return np.memmap(path, dtype=dtype, mode=open_mode, shape=shape,)
   

if __name__ == '__main__':
    dataloader = ImageDataset('MRI', 'mri_dg', '/projectnb/ivc-ml/dlteif/ayan_datasets/ADNI1/',
                                           'ADNI', 1, transform=imgutil.image_test('MRI'), test_envs=[1,2,3], filename='/projectnb/ivc-ml/dlteif/ADNI_NC_MCI_AD.csv', num_classes=3)
    
    for idx, data in enumerate(dataloader):
        ic(type(data), len(data))
        ic(data[0])
        ic(data[1].float().size())
        ic(data[2].long().size())
        # x = torch.cat([data[1] for data in minibatches])
        # y = torch.cat([data[2] for data in minibatches])
        # ic(x.size(), y.size())
    