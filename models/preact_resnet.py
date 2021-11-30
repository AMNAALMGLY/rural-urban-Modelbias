#adapted from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models._internally_replaced_utils import load_state_dict_from_url
from configs import args
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",

}
#TODO add downsample attribute
class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, in_channels,num_blocks, num_classes=1000):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        print(out.shape)
        #out = F.avg_pool2d(out, 4)
        out=self.avgpool(out)
        print(out.shape)
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out


def PreActResNet18(in_channels,pretrained):
    model = PreActResNet(PreActBlock, in_channels,[2,2,2,2],)
    if pretrained:
        state_dict = np.load(args.imagenet_weight_path) #TODO put it in urls list as in resnet
        new_dict=OrderedDict()
        for i,value in state_dict.items():
            print(i ,value)
        d= load_state_dict_from_url(model_urls['resnet18'])
        print(d)
        state_dict['conv0/W:0'] = nn.Parameter(
            init_first_layer_weights(in_channels, state_dict['conv0/W:0'], args.hs_weight_init))
        model.load_state_dict(state_dict)
    return model



def PreActResNet34(in_channels,pretrained):
    model=PreActResNet(PreActBlock, in_channels,[3,4,6,3])
    if pretrained:
        state_dict = np.load(args.imagenet_weight_path) #TODO put it in urls list as in resnet
        state_dict['conv0/W:0'] = nn.Parameter(
            init_first_layer_weights(in_channels, state_dict['conv0/W:0'], args.hs_weight_init))
        model.load_state_dict(state_dict)
    return model


def PreActResNet50(in_channels,pretrained):
    model=PreActResNet(PreActBottleneck,in_channels, [3,4,6,3])
    if pretrained:
        state_dict = np.load(args.imagenet_weight_path) #TODO put it in urls list as in resnet
        state_dict['conv1.weight'] = nn.Parameter(
            init_first_layer_weights(in_channels, state_dict['conv0/W:0'], args.hs_weight_init))
        model.load_state_dict(state_dict)
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
