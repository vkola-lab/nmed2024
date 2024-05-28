#%%
from time import time
from copy import deepcopy
import pandas as pd
import numpy as np
import toml
from tqdm import tqdm
# from adrd.nn import ImageModel
from icecream import ic
import torch
import ast
import os
import glob
import nibabel as nib
import json
#%%


def get_train_split(df, random_state, labels):
    if labels == 3:
        nc = df[df['NC'] == 1][:int(0.8 * len(df[df['NC'] == 1]))]
        mci = df[df['MCI'] == 1][:int(0.8 * len(df[df['MCI'] == 1]))]
        de = df[df['DE'] == 1][:int(0.8 * len(df[df['DE'] == 1]))]
        df_train = pd.concat([nc, mci, de], ignore_index=True)
    else:
        nc = df[df['NC'] == 1][:int(0.8 * len(df[df['NC'] == 1]))]
        mci = df[df['MCI'] == 1][:int(0.8 * len(df[df['MCI'] == 1]))]
        ad = df[df['AD'] == 1][:int(0.8 * len(df[df['AD'] == 1]))]
        lbd = df[df['LBD'] == 1][:int(0.8 * len(df[df['LBD'] == 1]))]
        vd = df[df['VD'] == 1][:int(0.8 * len(df[df['VD'] == 1]))]
        prion = df[df['PRD'] == 1]
        ftd = df[df['FTD'] == 1][:int(0.8 * len(df[df['FTD'] == 1]))]
        nph = df[df['NPH'] == 1][:int(0.8 * len(df[df['NPH'] == 1]))]
        # sef = df[df['SEF'] == 1][:int(0.8 * len(df[df['SEF'] == 1]))]
        psy = df[df['PSY'] == 1][:int(0.6 * len(df[df['PSY'] == 1]))]
        tbi = df[df['TBI'] == 1][:int(0.8 * len(df[df['TBI'] == 1]))]
        oc = df[df['ODE'] == 1][:int(0.6 * len(df[df['ODE'] == 1]))]

        df_train = pd.concat([nc, mci, ad, lbd, vd, prion, ftd, nph, psy, tbi, oc], ignore_index=True)
        
    df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_train.drop_duplicates(inplace=True)
    return df_train

def get_vld_split(df, random_state, df_train):
    df_vld = df[~df['ID'].isin(list(df_train['ID']))]
    df_vld = df_vld.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_vld.drop_duplicates(inplace=True)
    return df_vld

#%%
class CSVDataset:

    def __init__(self, dat_file, label_names=None, mri_emb_dict={}):
        ''' ... '''
        #%%
        # load data csv
        if isinstance(dat_file, str):
            df = pd.read_csv(dat_file)
        else:
            df = dat_file
            
        if 'mri_zip' in df.columns:
            print(len(df[~df['mri_zip'].isna()]))
            
        self.label_names = label_names
        
        def apply_mri_zip(row):
            if 'mri_zip' in row:
                if pd.isna(row['mri_zip']):
                    return row['ID']
                else:
                    return row['mri_zip']
            else:
                return row['ID']
            
        df['mri_zip'] = df.apply(apply_mri_zip, axis=1)
        df['filename_img'] = df['mri_zip'].map(mri_emb_dict)
        df.to_csv('./densenet_check.csv')
        
        print(f"{len(df[~df['filename_img'].isna()])} patients with MRI found")
        # print(df[~df['filename_img'].isna()]['filename_img'])
        
        def fill_nan_features(row):
            if isinstance(row['filename_img'], list):
                return pd.Series(row['filename_img'])
            else:
                return pd.Series([np.NaN] * 150)

        split_features = df.apply(fill_nan_features, axis=1)
        split_features.columns = [f"img_MRI_{i+1}" for i in range(len(split_features.columns))]
        df = pd.concat([df, split_features], axis=1)

        df = df.dropna(axis=1, how='all')
        all_labels = ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']
        all_labels = [lab for lab in all_labels if lab in df.columns]
        all_columns = ['ID'] + all_labels + [col for col in df.columns if col.startswith('img_MRI')]
        df = df[~df['filename_img'].isna()][all_columns]

        df.reset_index(drop=True, inplace=True)

        # drop rows where ALL labels are missing
        df_lbl = df[self.label_names]
        df_lbl = df_lbl.dropna(how='all')
        print('Out of {} samples, {} are dropped due to complete label missing.'.format(len(df), len(df) - len(df_lbl)))
        df = df[df.index.isin(df_lbl.index)]
        df.reset_index(drop=True, inplace=True)

        # change np.nan to None
        df.replace({np.nan: None}, inplace=True)
        self.df = df
        self.df.to_csv('./densenet_check.csv')
        # print(self.df.columns)
        columns = ['ID'] + self.label_names + [col for col in df.columns if col.startswith('img_MRI')]
        self.train = get_train_split(self.df, random_state=0, labels=len(self.label_names))[columns]
        self.vld = get_vld_split(self.df, random_state=0, df_train=self.train)[columns]
        print(set(self.train['ID']).intersection(set(self.vld['ID'])))
        
        print(len(self.train['ID']) + len(set(self.vld['ID'])))
        # self.tst = get_tst_split(df, 0, self.train, self.vld)
        
        # print(len(train))
        # print(len(vld))
        # print(len(tst))
        
        # lab_dict = {}
        # trn_lab_dict = {}
        # vld_lab_dict = {}
        # # tst_lab_dict = {}
        # for label in ['NC', 'MCI', 'DE', 'AD', 'LBD', 'VD', 'PRD', 'FTD', 'NPH', 'SEF', 'PSY', 'TBI', 'ODE']:
        #     lab_dict[label] = df[label].value_counts()
        #     trn_lab_dict[label] = train[label].value_counts()
        #     vld_lab_dict[label] = vld[label].value_counts()
        #     # tst_lab_dict[label] = tst[label].value_counts()
        # train_ratios = {k:(v[1] / lab_dict[k][1]) if (1 in v and 1 in lab_dict[k]) else 0 for k, v in trn_lab_dict.items()}
        # vld_ratios = {k:(v[1] / lab_dict[k][1]) if (1 in v and 1 in lab_dict[k]) else 0  for k, v in vld_lab_dict.items()}
        # test_ratios = {k:(v[1] / lab_dict[k][1]) if (1 in v and 1 in lab_dict[k]) else 0  for k, v in tst_lab_dict.items()}
        
        # # %%
        # train_ratios
        # # %%
        # vld_ratios
        # # %%
        # test_ratios
        
        
        # getting label fractions
        self.label_fractions = {}
        for label in self.label_names:
            # print(label)
            # print(self.df[label].value_counts()[1])
            try:
                self.label_fractions[label] = self.df[label].value_counts()[1] / len(self.df)
            except:
                self.label_fractions[label] = 0.3
                
    
    def get_features_labels(self, mode, return_dicts=True):
        #%%
        if mode == 0:
            df = self.train
        elif mode == 1:
            df = self.vld
        # elif mode == 2:
        #     df = self.tst
        else:
            df = self.df
            
        # self.ids = list(df['ID'])
        print(len(df))
        img_columns = df.columns[df.columns.str.startswith('img_MRI')]
        label_columns = self.label_names

        # Use melt to unpivot img columns into separate rows while keeping labels
        df = pd.melt(df, id_vars=label_columns, value_vars=img_columns, var_name='img_num', value_name='img_MRI')

        # Drop rows with NaN in the 'img' column
        df = df.dropna(subset=['img_MRI'])

        # Reset the index if needed
        df.reset_index(drop=True, inplace=True)
        
        print(len(df))
        
        # construct dictionaries for features and labels
        self.features, self.labels = [], []
        for i in range(len(df)):
            # print(df.iloc[i]['img_MRI'])
            self.features.append(df.iloc[i]['img_MRI'])
            self.labels.append(dict(zip(self.label_names, df.iloc[i][self.label_names])))
            # self.features[-1]['img_mode'] = img_mode
            
        for label in self.label_names:
            print(label, dict(df[label].value_counts()))
        
        if return_dicts:
            return [{'image':f, 'label':l} for (f,l) in zip(self.features, self.labels)], df
        
        return self.features, self.labels, df

    def __len__(self):
        ''' ... '''
        return len(self.df)

    # def __getitem__(self, idx):
    #     ''' ... '''
    #     return self.features[idx], self.labels[idx]


if __name__ == '__main__':
    # load dataset
    dset = CSVDataset(mode=0, split=[8, 2])
    print(dset[0])

# %%
