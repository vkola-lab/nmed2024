#%%
import pandas as pd
import numpy as np
import toml
import datetime
import torch
import random
import ast
import os
import glob
import nibabel as nib
import json

from tqdm import tqdm
from .MRI_load import load_mris
from collections import defaultdict
from time import time
from copy import deepcopy
from icecream import ic
# from adrd.nn import ImageModel
#%%


value_mapping = {
    'his_SEX':          {'female': 0, 'male': 1},
    'his_HISPANIC':     {'no': 0, 'yes': 1},
    'his_NACCNIHR':     {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'mul': 5},
    'his_RACE':         {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'oth': 5},
    'his_RACESEC':      {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'oth': 5},
    'his_RACETER':      {'whi': 0, 'blk': 1, 'asi': 2, 'ind': 3, 'haw': 4, 'oth': 5},
}

class CSVDataset:

    def __init__(self, dat_file, cnf_file, mode=0, img_mode=0, arch=None, transforms=None, stripped=None):
        ''' ... '''
        # load data csv
        if isinstance(dat_file, str):
            print(dat_file)
            df = pd.read_csv(dat_file)
        else:
            df = dat_file
            
        # load configuration file
        self.cnf = toml.load(cnf_file)
        
        if 'ID' in df.columns:
            self.ids = list(df['ID'])

        df.reset_index(drop=True, inplace=True)
        print('{} are selected for mode {}.'.format(len(df), mode))

        # check feature availability in data file
        print('Out of {} features in configuration file, '.format(len(self.cnf['feature'])), end='')
        tmp = [fea for fea in self.cnf['feature'] if fea not in df.columns]
        print('{} are unavailable in data file.'.format(tmp))

        # check label availability in data file
        print('Out of {} labels in configuration file, '.format(len(self.cnf['label'])), end='')
        tmp = [lbl for lbl in self.cnf['label'] if lbl not in df.columns]
        print('{} are unavailable in data file.'.format(len(tmp)))

        self.cnf['feature'] = {k:v for k,v in self.cnf['feature'].items() if k in df.columns}
        self.cnf['label'] = {k:v for k,v in self.cnf['label'].items() if k in df.columns}

        # get feature and label names
        features = list(self.cnf['feature'].keys())
        labels = list(self.cnf['label'].keys())

        # omit features that are not present in dat_file
        features = [fea for fea in features if fea in df.columns]

        # mri
        img_fea_to_pop = []
        total = 0
        total_cohorts = {}
        for fea in self.cnf['feature'].keys():
            # print('fea: ', fea)
            if self.cnf['feature'][fea]['type'] == 'imaging':
                print('imaging..')
                if img_mode == -1:
                    # to train non imaging model
                    img_fea_to_pop.append(fea)
                elif img_mode == 0:
                    print("fea: ", fea)
                    filenames = df[fea].dropna().to_list()
                    with open('notexists.txt', 'a') as f:
                        for fn in filenames:
                            if fn is None:
                                continue
                            if not os.path.exists(fn):
                                f.write(fn+'\n')
                            
                            mri_name = fn.split('/')[-1]
                            total += 1

                # load MRI embeddings 
                elif img_mode == 1:
                    print("fea: ", fea)
                    filenames = df[fea].to_list()
                    # print(len(filenames))
                    
                    if len(df[~df[fea].isna()]) == 0:
                        continue
                    # print(fea)
                        
                    npy = []
                    n = 0
                    for fn in tqdm(filenames):
                        try:
                            # print('fn: ', fn)
                            data = np.load(fn, mmap_mode='r')
                            if np.isnan(data).any():
                                npy.append(None)
                                continue
                            
                            if 'swinunet' in fn.lower():
                                if len(data.shape) < 5:
                                    data = np.expand_dims(data, axis=0)
                                    
                            npy.append(data)

                            self.cnf['feature'][fea]['shape'] = data.shape
                            self.cnf['feature'][fea]['img_shape'] = data.shape
                            
                            
                            # print(data.shape)
                            n += 1
                        except:
                            npy.append(None)
                    # print(self.cnf['feature'][fea]['shape'])
                    print(f"{n} MRI embeddings found with shape {self.cnf['feature'][fea]['shape']}")
                    
                    total += n
                    print(len(df), len(npy))
                    df[fea] = npy
                    # return

                elif img_mode == 2: 
                    # load MRIs and use swinunetr model to get the embeddings
                    print('img_mode is 2')
                    embedding_dict = load_mris.get_emb('filename', df, arch=arch, transforms=transforms, stripped=stripped)
                    mri_embeddings = []
                    for index, row in df.iterrows():
                        filename = row['filename']
                        print(filename)
                        if filename in embedding_dict:

                            emb = embedding_dict[filename].flatten()
                            mri_embeddings.append(emb)
                            self.cnf['feature'][fea]['shape'] = emb.shape
                            self.cnf['feature'][fea]['img_shape'] = emb.shape
                        else:
                            mri_embeddings.append(None)
                    print(avail)

                    df[fea] = mri_embeddings
                    if 'img_shape' in self.cnf['feature'][fea]:
                        print(self.cnf['feature'][fea]['img_shape'])

        print(f"Total mri embeddings found: {total}")

        for fea in img_fea_to_pop:
            self.cnf['feature'].pop(fea)

        df = df.drop(img_fea_to_pop, axis=1)
        features = [fea for fea in features if fea in df.columns]
        labels = [lab for lab in labels if lab in df.columns]

        # drop columns that are not present in configuration
        df = df[features + labels]

        # drop rows where ALL features are missing
        df_fea = df[features]
        df_fea = df_fea.dropna(how='all')
        print('Out of {} samples, {} are dropped due to complete feature missing.'.format(len(df), len(df) - len(df_fea)))
        df = df[df.index.isin(df_fea.index)]
        df.reset_index(drop=True, inplace=True)

        # drop rows where ALL labels are missing
        df_lbl = df[labels]
        df_lbl = df_lbl.dropna(how='all')
        print('Out of {} samples, {} are dropped due to complete label missing.'.format(len(df), len(df) - len(df_lbl)))
        df = df[df.index.isin(df_lbl.index)]
        df.reset_index(drop=True, inplace=True)
        

        # some of the values need to be mapped to the desirable domain
        for name in features + labels:
            if name in value_mapping:
                col = df[name].to_list()
                try:
                    col = [value_mapping[name][s] if not pd.isnull(s) else None for s in col]
                except KeyError as err:
                    print(err, name)
                    exit()
                df[name] = col
                
        # print(features)
        

        # change np.nan to None
        df.replace({np.nan: None}, inplace=True)

        
        # done for df
        self.df = df

        # construct dictionaries for features and labels
        self.features, self.labels = [], []
        keys = df.columns.values.tolist()
        for i in range(len(df)):
            vals = df.iloc[i].to_list()
            self.features.append(dict(zip(keys[:len(features)], vals[:len(features)])))
            self.labels.append(dict(zip(keys[len(features):], vals[len(features):])))
        
        # test: remove if None
        for i in range(len(self.features)):
            for k, v in list(self.features[i].items()):
                if v is None:
                    self.features[i].pop(k)


        # getting label fractions
        self.label_fractions = {}
        for label in labels:
            try:
                self.label_fractions[label] = self.df[label].value_counts()[1] / len(self.df)
            except:
                self.label_fractions[label] = 0.3

    def __len__(self):
        ''' ... '''
        return len(self.df)

    def __getitem__(self, idx):
        ''' ... '''
        return self.features[idx], self.labels[idx]

    def _get_mask_mode(self, df, mode, split, seed):
        ''' ... '''
        # normalize split into ratio
        ratio = np.array(split) / np.sum(split)
        
        # list of modes for all samples
        arr = []
        
        # 0th ~ (N-1)th modes
        for i in range(len(split) - 1):
            arr += [i] * round(ratio[i] * len(df))
        
        # last mode
        arr += [len(split) - 1] * (len(df) - len(arr))
        
        # random shuffle
        # random seed will be fixed before shuffle and reset right after
        arr = np.array(arr)
        np.random.seed(seed)
        np.random.shuffle(arr)
        np.random.seed(int(1000 * time()) % 2 ** 32)
        
        # generate mask
        msk = (arr == mode).tolist()
        
        return msk
    
    @property
    def feature_modalities(self):
        ''' ... '''
        return self.cnf['feature']

    @property
    def label_modalities(self):
        ''' ... '''
        return self.cnf['label']


if __name__ == '__main__':
    # load dataset
    dset = CSVDataset(mode=0, split=[8, 2])
    print(dset[0])

# %%
