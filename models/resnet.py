# resnet:credits:https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

from typing import Type, Any, Callable, Union, List, Optional

import torch
from torch import Tensor

import torch.nn as nn
from models._internally_replaced_utils import load_state_dict_from_url
from models.utils import _log_api_usage_once
from configs import args
import time
import pytorch_lightning

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",

]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    # "resnet50":'https://zenodo.org/record/4728033/files/seco_resnet50_100k.ckpt?download=1',
    'resnet50': "https://download.pytorch.org/models/resnet50-0676ba61.pth",

}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


'''
class Self_Attn(nn.Module):
    """ Self attention Layer
        https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """

    def __init__(self, in_dim, activation="relu"):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention
'''


class SE_Block(nn.Module):
    "credits: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py#L4"

    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        #self.se = SE_Block(c=planes)

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        #out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the strhttps://zenodo.org/record/4728033/files/seco_resnet50_100k.ckpt?download=1ide at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            in_channels: int,
            layers: List[int],

            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        # Attention
        self.attn = SE_Block(c=512)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            planes: int,
            blocks: int,
            stride: int = 1,
            dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) :
        # See note [TorchScript super()]

        # x,_=self.attn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # x,_=self.attn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # print('before attention',x.shape)
        #x=self.attn(x)

        # print('after attention',x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = x
        x = self.fc(x)

        return x,features

    def forward(self, x: Tensor) :
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        in_channels: int,
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any,
) -> ResNet:
    model = ResNet(block, in_channels, layers, **kwargs)
    if pretrained:
        # state_dict=torch.load('seco_resnet50_100k.ckpt')
        # if using attention:
        # attn_weights=["attn.gamma", "attn.query_conv.weight", "attn.query_conv.bias", "attn.key_conv.weight",
        #             "attn.key_conv.bias", "attn.value_conv.weight", "attn.value_conv.bias"]
        #attn_weights=["attn.excitation.0.weight","attn.excitation.2.weight" ]
        """
        attn_weights = [
                        "layer1.0.se.excitation.0.weight", "layer1.0.se.excitation.2.weight",
                        "layer1.1.se.excitation.0.weight",
                        "layer1.1.se.excitation.2.weight", "layer2.0.se.excitation.0.weight",
                        "layer2.0.se.excitation.2.weight", "layer2.1.se.excitation.0.weight",
                        "layer2.1.se.excitation.2.weight", "layer3.0.se.excitation.0.weight",
                        "layer3.0.se.excitation.2.weight", "layer3.1.se.excitation.0.weight",
                        "layer3.1.se.excitation.2.weight", "layer4.0.se.excitation.0.weight",
                        "layer4.0.se.excitation.2.weight", "layer4.1.se.excitation.0.weight",
                        "layer4.1.se.excitation.2.weight"]
        """
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        state_dict['conv1.weight'] = nn.Parameter(
            init_first_layer_weights(in_channels, state_dict['conv1.weight'], args.hs_weight_init))
        # print(model.state_dict())
        # if 'attn' in model.state_dict():
        '''
        for key in attn_weights:
            if 'weight' in key:

                nn.init.kaiming_normal_(model.state_dict()[key], mode="fan_out", nonlinearity="relu")
            else:
                nn.init.constant_(model.state_dict()[key], 0.0)
            state_dict[key] = model.state_dict()[key]

        model.load_state_dict(state_dict)
        '''
    return model


def resnet18(in_channels: int, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, in_channels, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(in_channels: int, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, in_channels, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(in_channels: int, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, in_channels, [3, 4, 6, 3], pretrained, progress, **kwargs)


def init_first_layer_weights(in_channels: int, rgb_weights,
                             hs_weight_init: str):
    '''Initializes the weights for filters in the first conv layer.

      If we are using RGB-only, then just initializes var to rgb_weights. Otherwise, uses
      hs_weight_init to determine how to initialize the weights for non-RGB bands.

      Args
      - int: in_channesl, input channels
          - in_channesl is  either 3 (RGB), 7 (lxv3), or 9 (Landsat7) or 2 (NL)
      - rgb_weights: ndarray of np.float32, shape [64, 3, F, F]
      - hs_weight_init: str, one of ['random', 'same', 'samescaled']

      Returs
      -torch tensor : final_weights
      '''

    out_channels, rgb_channels, H, W = rgb_weights.shape
    print('rgb weight shape ', rgb_weights.shape)
    rgb_weights = torch.tensor(rgb_weights)
    ms_channels = in_channels - rgb_channels
    if in_channels == 3:
        if args.include_buildings:
            with torch.no_grad():
                mean = rgb_weights.mean()
                std = rgb_weights.std()
                final_weights = torch.empty((out_channels, in_channels, H, W))
                final_weights = torch.nn.init.trunc_normal_(final_weights, mean, std)
        else:
            final_weights = rgb_weights

    elif in_channels < 3:  # NL
        with torch.no_grad():
            mean = rgb_weights.mean()
            std = rgb_weights.std()
            final_weights = torch.empty((out_channels, in_channels, H, W))
            final_weights = torch.nn.init.trunc_normal_(final_weights, mean, std)
    elif in_channels > 3:
        # spectral images

        if hs_weight_init == 'same':

            with torch.no_grad():
                mean = rgb_weights.mean(dim=1, keepdim=True)  # mean across the in_channel dimension
                mean = torch.tile(mean, (1, ms_channels, 1, 1))
                ms_weights = mean

        elif hs_weight_init == 'random':
            start = time.time()
            with torch.no_grad():
                mean = rgb_weights.mean()
                std = rgb_weights.std()
                ms_weights = torch.empty((out_channels, ms_channels, H, W))
                ms_weights = torch.nn.init.trunc_normal_(ms_weights, mean, std)
            print(f'random: {time.time() - start}')

        elif hs_weight_init == 'samescaled':
            start = time.time()
            with torch.no_grad():
                mean = rgb_weights.mean(dim=1, keepdim=True)  # mean across the in_channel dimension
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

class MLP(nn.Module):
   def __init__(self, input_dim, output_dim=512):
      super().__init__()
      self.input_dim=input_dim
      self.output_dim=output_dim
      self.layer1=nn.Linear(input_dim,output_dim *2)
      self.layer2=nn.Linear(output_dim *2,output_dim)
      #self.layer3=nn.Linear(output_dim//4 , output_dim)
      self.fc=nn.Linear(output_dim,1)
      #self.layer3=nn.Linear(output_dim,1)
      self.relu=nn.ReLU()
   def forward(self,x):
       return self.fc(self.relu(self.layer2( self.relu(self.layer1(x))))),self.layer2( self.relu(self.layer1(x)))

def mlp(in_channels: int, pretrained: bool = False, progress: bool = True, **kwargs: Any) :
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """


    return MLP(in_channels)
