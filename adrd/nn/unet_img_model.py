from pyexpat import features
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import numpy as np
import re
from icecream import ic
import math
import torch.nn.utils.weight_norm as weightNorm

# from . import UNet3DBase
from .unet_3d import UNet3DBase


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)

class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        # if type in ['conv', 'gap'] and len(bottleneck_dim) > 3:
            # bottleneck_dim = bottleneck_dim[-3:]
        ic(bottleneck_dim)
        if type == 'wn':
            self.layer = weightNorm(
                nn.Linear(bottleneck_dim[1:], class_num), name="weight")
            # self.fc.apply(init_weights)
        elif type == 'gap':
            if len(bottleneck_dim) > 3:
                bottleneck_dim = bottleneck_dim[-3:]
            self.layer = nn.AvgPool3d(bottleneck_dim, stride=(1,1,1))
        elif type == 'conv':
            if len(bottleneck_dim) > 3:
                bottleneck_dim = bottleneck_dim[-4:]
            ic(bottleneck_dim)
            self.layer = nn.Conv3d(bottleneck_dim[0], class_num, kernel_size=bottleneck_dim[1:])
            ic(self.layer)
        else:
            print('bottleneck dim: ', bottleneck_dim)
            self.layer = nn.Sequential(
                            torch.nn.Flatten(start_dim=1, end_dim=-1),
                            nn.Linear(math.prod(bottleneck_dim), class_num)
            )
        self.layer.apply(init_weights)

    def forward(self, x):
        # print('=> feat_classifier forward')
        # ic(x.size())
        x = self.layer(x)
        # ic(x.size())
        if self.type in ['gap','conv']:
            x = torch.squeeze(x)
            if len(x.shape) < 2:
                x = torch.unsqueeze(x,0)
        # print('returning x: ', x.size())
        return x

class ImageModel(nn.Module):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(
            self, 
            counts=None,
            classifier='gap',
            accum_iter=8,
            save_emb=False,
            # ssl,
            num_classes=1,
            load_img_ckpt=False,
        ):
        super(ImageModel, self).__init__()
        if counts is not None:
            if isinstance(counts[0], list):
                counts = np.stack(counts, axis=0).sum(axis=0)
                print('counts: ', counts)
                total = np.sum(counts)
                print(total/counts)
                self.weight = total/torch.FloatTensor(counts)
            else:
                total = sum(counts)
                self.weight = torch.FloatTensor([total/c for c in counts])
        else:
            self.weight = None
        print('weight: ', self.weight)
        # device = torch.device(f'cuda:{args.gpu_id}' if args.gpu_id is not None else 'cpu')
        self.criterion = nn.CrossEntropyLoss(weight=self.weight)
        # if ssl:
        #     # add contrastive loss
        #     # self.ssl_criterion = 
        #     pass

        self.featurizer = UNet3DBase(n_class=num_classes, attention=True, pretrained=load_img_ckpt)
        self.classifier = feat_classifier(
            num_classes, self.featurizer.in_features, classifier)

        self.network = nn.Sequential(
            self.featurizer, self.classifier)
        self.accum_iter = accum_iter
        self.acc_steps = 0
        self.save_embedding = save_emb

    def update(self, minibatches, opt, sch, scaler):
        print('--------------def update----------------')
        device = list(self.parameters())[0].device
        all_x = torch.cat([data[1].to(device).float() for data in minibatches])
        all_y = torch.cat([data[2].to(device).long() for data in minibatches])
        print('all_x: ', all_x.size())
        # all_p = self.predict(all_x)
        # all_probs =  
        label_list = all_y.tolist()
        count = float(len(label_list))
        ic(count)
            
        uniques = sorted(list(set(label_list)))
        ic(uniques)
        counts = [float(label_list.count(i)) for i in uniques]
        ic(counts)
        
        weights = [count / c for c in counts]
        ic(weights)
        
        with autocast():
            loss = self.criterion(self.predict(all_x), all_y)
        self.acc_steps += 1
        print('class: ', loss.item())

        scaler.scale(loss / self.accum_iter).backward()
        
        if self.acc_steps == self.accum_iter:
            scaler.step(opt)
            if sch:
                sch.step()
            scaler.update()
            self.zero_grad()
            self.acc_steps = 0
            torch.cuda.empty_cache()
            
        del all_x
        del all_y
        return {'class': loss.item()}, sch

    def forward(self, *args, **kwargs):
        return self.network(*args, **kwargs)
    
    def predict(self, x, stage='normal', attention=False):
        # print('network device: ', list(self.network.parameters())[0].device)
        # print('x device: ', x.device)
        if stage == 'get_features' or self.save_embedding:
            feats = self.network[0](x, attention=attention)
            output = self.network[1](feats[-1] if attention else feats)
            return feats, output
        else:
            return self.network(x)

    def extract_features(self, x, attention=False):
        feats = self.network[0](x, attention=attention)
        return feats

    def load_checkpoint(self, state_dict):
        try:
            self.load_checkpoint_helper(state_dict)
        except:
            featurizer_dict = {}
            net_dict = {}
            for key,val in state_dict.items():
                if 'featurizer' in key:
                    featurizer_dict[key] = val
                elif 'network' in key:
                    net_dict[key] = val
            self.featurizer.load_state_dict(featurizer_dict)
            self.classifier.load_state_dict(net_dict)

    def load_checkpoint_helper(self, state_dict):
        try:
            self.load_state_dict(state_dict)
            print('try: loaded')
        except RuntimeError as e:
            print('--> except')
            if 'Missing key(s) in state_dict:' in str(e):
                state_dict = {
                    key.replace('module.', '', 1): value
                    for key, value in state_dict.items()
                }
                state_dict = {
                    key.replace('featurizer.', '', 1).replace('classifier.','',1): value
                    for key, value in state_dict.items()
                }
                state_dict = {
                    re.sub('network.[0-9].', '', key): value
                    for key, value in state_dict.items()
                }
                try:
                    del state_dict['criterion.weight']
                except:
                    pass
                self.load_state_dict(state_dict)
                
                print('except: loaded')