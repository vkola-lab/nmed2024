import torch
import torch.nn as nn
import sys
from icecream import ic
# sys.path.append('/home/skowshik/ADRD_repo/adrd_tool/adrd/')
from .net_resnet3d import r3d_18
# from dev.data.dataset_csv import CSVDataset


class ResNetModel(nn.Module):
    ''' ... '''
    def __init__(
        self, 
        tgt_modalities,
        mri_feature = 'img_MRI_T1',
    ):
        ''' ... '''
        super().__init__()

        self.mri_feature = mri_feature

        self.img_net_ = r3d_18()

        # self.modules_emb_src = nn.Sequential(
        #         nn.BatchNorm1d(9),
        #         nn.Linear(9, d_model)
        #     )

        # classifiers (binary only)
        self.modules_cls = nn.ModuleDict()
        for k, info in tgt_modalities.items():
            if info['type'] == 'categorical' and info['num_categories'] == 2:
                # categorical
                self.modules_cls[k] = nn.Linear(64, 1)

            else:
                # unrecognized
                raise ValueError

    def forward(self, x):
        ''' ... '''
        tgt_iter = self.modules_cls.keys()

        img_x_batch = x[self.mri_feature]
        img_out = self.img_net_(img_x_batch)

        # ic(img_out.shape)

        # run linear classifiers
        out = [self.modules_cls[k](img_out).squeeze(1) for i, k in enumerate(tgt_iter)]
        out = torch.stack(out, dim=1)

        # ic(out.shape)

        # out to dict
        out = {k: out[:, i] for i, k in enumerate(tgt_iter)}
            
        return out


if __name__ == '__main__':
    ''' for testing purpose only '''
    # import torch
    # import numpy as np

    # seed = 0
    # print('Loading training dataset ... ')
    # dat_trn = CSVDataset(mode=0, split=[1, 700], seed=seed)
    # print(len(dat_trn))
    # tgt_modalities = dat_trn.label_modalities
    # net = ResNetModel(tgt_modalities).to('cuda')
    # x = dat_trn.features
    # x = {k: torch.as_tensor(np.array([x[i][k] for i in range(len(x))])).to('cuda') for k in x[0]}
    # ic(x)
    

    # # print(net(x).shape)
    # print(net(x))



