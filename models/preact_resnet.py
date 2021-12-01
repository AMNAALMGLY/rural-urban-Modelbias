#adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
import re
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models._internally_replaced_utils import load_state_dict_from_url
from configs import args
from models.resnet import model_urls


def load_tensor_pack(model,path,in_channels):
    tensor_pack_dict = np.load(path)  # tensor pack dict
    my_dict = model.state_dict().copy()  # torch model dict copy
    state_dict = model.state_dict()  # torch model dict reference
    running = dict()  # running mean dict
    EMA=dict()   #EMA dict from tensorpack

    # put keys of running mean into a new dict
    # del keys that are not in the tensor pack(track_num_batches)
    for key, value in state_dict.items():
        if 'running' in key:
            running[key] = value
            del my_dict[key]
        if 'batches' in key:
            del my_dict[key]
    # assign values of tensor packs to model dict orderly
    for key1, value2 in zip(my_dict.keys(), tensor_pack_dict.values()):
        my_dict[key1] = value2
    # del all keys that are not running mean from tensorpack
    for key in tensor_pack_dict.keys():
        if 'EMA' not in key:
            continue
        else:
            EMA[key]=tensor_pack_dict[key]

    # assign values of the edited tensorpack to keys of running dict
    for key1, value2 in zip(running.keys(), EMA.values()):
        running[key1] = value2
    # load values into models state_dict
    for key in state_dict.keys():
        if 'running' in key:
            state_dict[key] = torch.tensor(running[key],requires_grad=True)
        elif 'running'  not in key:
            if 'num_batches' not in key:

                  state_dict[key] = torch.tensor(my_dict[key],requires_grad=True)
                  if 'conv' in key :
                      state_dict[key]=state_dict[key].reshape(state_dict[key].shape[-1],state_dict[key].shape[-2],state_dict[key].shape[-3],state_dict[key].shape[-4])
    state_dict['conv1.weight']=nn.Parameter(
            init_first_layer_weights(in_channels, state_dict['conv1.weight'], args.hs_weight_init),requires_grad=True)

    return model

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, preactivate=True):
        super(PreActBlock, self).__init__()
        self.pre_bn = self.pre_relu = None
        if preactivate:
            self.pre_bn = nn.BatchNorm2d(inplanes)
            self.pre_relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1_2 = nn.BatchNorm2d(planes)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride
        self.preactivate = preactivate

    def forward(self, x):
        if self.preactivate:
            preact = self.pre_bn(x)
            preact = self.pre_relu(preact)
        else:
            preact = x

        out = self.conv1(preact)
        out = self.bn1_2(out)
        out = self.relu1_2(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(preact)
        else:
            residual = x

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, preactivate=True):
        super(PreActBottleneck, self).__init__()
        self.pre_bn = self.pre_relu = None
        if preactivate:
            self.pre_bn = nn.BatchNorm2d(inplanes)
            self.pre_relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(planes)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(planes)
        self.relu2_3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.preactivate = preactivate

    def forward(self, x):
        if self.preactivate:
            preact = self.pre_bn(x)
            preact = self.pre_relu(preact)
        else:
            preact = x

        out = self.conv1(preact)
        out = self.bn1_2(out)
        out = self.relu1_2(out)

        out = self.conv2(out)
        out = self.bn2_3(out)
        out = self.relu2_3(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(preact)
        else:
            residual = x

        out += residual

        return out

class PreActResNet(nn.Module):
    def __init__(self, block, in_channels,layers, num_classes=1000):
        self.inplanes = 64
        super(PreActResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.final_bn = nn.BatchNorm2d(512 * block.expansion)
        self.final_relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        # On the first residual block in the first residual layer we don't pre-activate,
        # because we take care of that (+ maxpool) after the initial conv layer
        preactivate_first = stride != 1

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, preactivate_first))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.final_bn(x)
        x = self.final_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def PreActResNet18(in_channels,pretrained):
    model = PreActResNet(PreActBlock, in_channels,[2,2,2,2],)
    if pretrained:
            model = load_tensor_pack(model, args.imagenet_weight_path, in_channels)
    print(model.state_dict()['conv1.weight'].requires_grad)


    return model


def PreActResNet34(in_channels,pretrained):
    model=PreActResNet(PreActBlock, in_channels,[3,4,6,3])
    #TODO edit load tensor pack function to adapt resnet34 and resnet50
    if pretrained:
       model=load_tensor_pack(model,args.imagenet_weight_path,in_channels)

    return model


def PreActResNet50(in_channels,pretrained):
    model=PreActResNet(PreActBottleneck,in_channels, [3,4,6,3])
    if pretrained:
        model = load_tensor_pack(model, args.imagenet_weight_path, in_channels)
    return model

'''
def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3,4,23,3])

def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3,8,36,3])
'''
def init_first_layer_weights(in_channels: int, rgb_weights,
                                 hs_weight_init: str) :
    out_channels, rgb_channels, H, W = rgb_weights.shape
    print('rgb weight shape ',rgb_weights.shape)
    rgb_weights=torch.tensor(rgb_weights)
    ms_channels = in_channels - rgb_channels
    if in_channels == 3:
        final_weights = rgb_weights
    elif in_channels <3:
        with torch.no_grad():
            mean = rgb_weights.mean()
            std = rgb_weights.std()
            final_weights = torch.normal(mean, std, size=(out_channels, in_channels, H, W))
    elif in_channels > 3:
        # spectral images

        if hs_weight_init == 'same':

            with torch.no_grad():
                mean = rgb_weights.mean(axis=1, keepdims=True)  # mean across the in_channel dimension
                mean = torch.tile(mean, (1, ms_channels, 1, 1))
                ms_weights = mean

        elif hs_weight_init == 'random':
            start = time.time()
            with torch.no_grad():
                mean = rgb_weights.mean()
                std = rgb_weights.std()
                ms_weights = torch.normal(mean, std, size=(out_channels, ms_channels, H, W))
            print(f'random: {time.time() - start}')

        elif hs_weight_init == 'samescaled':
            start = time.time()
            with torch.no_grad():
                mean = rgb_weights.mean(axis=1, keepdims=True)  # mean across the in_channel dimension
                mean = torch.tile(mean, (1, ms_channels, 1, 1))
                ms_weights = (mean * 3) / (3 + ms_channels)
                # scale both rgb_weights and ms_weights
                rgb_weights = (rgb_weights * 3) / (3 + ms_channels)
            print(f'samescaled: {time.time() - start}')

        else:

            raise ValueError(f'Unknown hs_weight_init type: {hs_weight_init}')

        final_weights = torch.cat([rgb_weights, ms_weights], dim=1)
    print('init__layer_weight shape ', final_weights.shape)
    return final_weights

def get_saved_var_name(model_var, bottleneck=False):
    '''Gets the saved variable's name (from TensorPack) for the given model variable.

    Args
    - model_var: tf.Variable
    - bottleneck: bool

    Returns: str, the saved variable name if the model variable is part of the ResNet,
        None, otherwise.
    '''
    saved_var_name = model_var.name

    # strip the variable name of everything up to and including '*resnet/'
    s = re.search(r'.*resnet/', saved_var_name)
    if s is None:  # if 'resnet/' not in variable name, then it isn't part of the resnet
        return None
    else:
        saved_var_name = saved_var_name.replace(s.group(0), '')

    # replace model variable names with the saved variable counterparts
    conversion = {
        'batch_normalization': 'bn',
        'weights:0': 'W:0',
        'shortcut': 'convshortcut',
        'scale1': 'conv0'  # 'scale1' is a special case, because it has no blocks or subblocks
    }

    if bottleneck:
        conversion['/a/'] = '/conv1/'
        conversion['/b/'] = '/conv2/'
        conversion['/c/'] = '/conv3/'
    else:
        conversion['/A/'] = '/conv1/'
        conversion['/B/'] = '/conv2/'

    for model_str, saved_str in conversion.items():
        saved_var_name = saved_var_name.replace(model_str, saved_str)

    # shift all of the 'block' numbers down by 1
    # search for 'block###', return the '###' part
    s = re.search(r'block(\d+)', saved_var_name)  # search for 'scale###'
    if s is not None:
        block_str = s.group(0)  # block_str = 'block###'
        block_num = int(s.group(1))  # extract the '###' part from 'scale###' and convert it to int
        new_block_str = 'block' + str(block_num - 1)
        saved_var_name = saved_var_name.replace(block_str, new_block_str)

    # shift all of the 'scale' numbers down by 2, then rename to 'group'
    # - NOTE: we already dealt with scale1 above since it is a special case (no blocks or subblocks)
    #   so we don't need to worry about negative numbers here
    s = re.search(r'scale(\d+)', saved_var_name)  # search for 'scale###'
    if s is not None:
        scale_str = s.group(0)  # scale_str = 'scale###'
        scale_num = int(s.group(1))  # extract the '###' part from 'scale###' and convert it to int
        new_group_str = 'group' + str(scale_num - 2)
        saved_var_name = saved_var_name.replace(scale_str, new_group_str)

    return saved_var_name
