# This implementation is based on the DenseNet-BC implementation in torchvision
# https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
# https://github.com/gpleiss/efficient_densenet_pytorch/blob/master/models/densenet.py
# https://github.com/vkola-lab/ncomms2022/blob/main/backends/DenseNet.py


import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate
        self.efficient = efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
            (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        tgt_modalities (list) - list of target modalities
        efficient (bool) - set to True to use checkpointing. Much more memory efficient, but slower.
    """
    # def __init__(self, tgt_modalities, growth_rate=12, block_config=(3, 3, 3), compression=0.5,
    #              num_init_features=16, bn_size=4, drop_rate=0, efficient=False, load_from_ckpt=False): # config 1
    
    def __init__(self, tgt_modalities, growth_rate=12, block_config=(3, 3, 3), compression=0.5,
                 num_init_features=16, bn_size=4, drop_rate=0, efficient=False, load_from_ckpt=False): # config 2
        
        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([('conv0', nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=0, bias=False)),]))
        self.features.add_module('norm0', nn.BatchNorm3d(num_init_features))
        self.features.add_module('relu0', nn.ReLU(inplace=True))
        self.features.add_module('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=0, ceil_mode=False))
        self.tgt_modalities = tgt_modalities

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                efficient=efficient,
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config):
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=int(num_features * compression))
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = int(num_features * compression)

        # Final batch norm
        self.features.add_module('norm_final', nn.BatchNorm3d(num_features))
        
        # Classification heads
        self.tgt = torch.nn.ModuleDict()
        for k in tgt_modalities:
            # self.tgt[k] = torch.nn.Linear(621, 1) # config 2
            self.tgt[k] = torch.nn.Sequential(
                    torch.nn.Linear(self.test_size(), 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 1)
                )

        print(f'load_from_ckpt: {load_from_ckpt}')
        # Initialization
        if not load_from_ckpt:
            for name, param in self.named_parameters():
                if 'conv' in name and 'weight' in name:
                    n = param.size(0) * param.size(2) * param.size(3) * param.size(4)
                    param.data.normal_().mul_(math.sqrt(2. / n))
                elif 'norm' in name and 'weight' in name:
                    param.data.fill_(1)
                elif 'norm' in name and 'bias' in name:
                    param.data.fill_(0)
                elif ('classifier' in name or 'tgt' in name) and 'bias' in name:
                    param.data.fill_(0)

        # self.size = self.test_size()

    def forward(self, x, shap=True):
        # print(x.shape)
        features = self.features(x)
        # print(features.shape)
        out = F.relu(features, inplace=True)
        # out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        
        # print(out.shape)
        
        # out_tgt = self.tgt(out).squeeze(1)
        # print(out_tgt)
        # return F.softmax(out_tgt)
        
        tgt_iter = self.tgt.keys()
        out_tgt = {k: self.tgt[k](out).squeeze(1) for k in tgt_iter}
        if shap:
            out_tgt = torch.stack(list(out_tgt.values()))
            return out_tgt.T
        else: 
            return out_tgt

    def test_size(self):
        case = torch.ones((1, 1, 182, 218, 182))
        output = self.features(case).view(-1).size(0)
        return output


if __name__ == "__main__":
    model = DenseNet(
        tgt_modalities=['NC', 'MCI', 'DE'], 
        growth_rate=12, 
        block_config=(2, 3, 2), 
        compression=0.5,
        num_init_features=16, 
        drop_rate=0.2)
    print(model)
    torch.manual_seed(42)
    x = torch.rand((1, 1, 182, 218, 182))
    # layers = list(model.features.named_children())
    features = nn.Sequential(*list(model.features.children()))(x)
    print(features.shape)
    print(sum(p.numel() for p in model.parameters()))
    # # out = mdl.net_(x, shap=False)
    # # print(out)

    # out = model(x, shap=False)
    # print(out)
    # layer_found = False
    # features = None
    # desired_layer_name = 'transition3'

    # for name, layer in layers:
    #     if name == desired_layer_name:
    #         x = layer(x)
    #         print(x)
    # model(x)
    # print(features)