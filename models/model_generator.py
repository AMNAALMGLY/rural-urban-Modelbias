import torch
from torch import nn

from configs import args
from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50
from models.resnet import resnet18, resnet34, resnet50, mlp
from utils.utils import load_from_checkpoint

model_type = dict(resnet18=resnet18,
                  resnet34=resnet34,
                  resnet50=resnet50,
                  mlp=mlp,
                  )


def get_model(model_name, in_channels, pretrained=False, ckpt_path=None):
    model_fn = model_type[model_name]

    model = model_fn(in_channels, pretrained)
    if ckpt_path:
        model = load_from_checkpoint(ckpt_path, model)
    return model


class Encoder(nn.Module):
    def __init__(self, resnet_bands=None, resnet_build=None, Mlp=None, dim=512, num_outputs=1):
        # TODO add resnet_NL and resnet_Ms
        # TODO add multiple mlps for metadata
        """
            Args:

                resnet_bands (nn.Module): Encoder layer of the self attention
                resnet_build (nn.Module): Encoder layer of intersample attention
                MLp (int): Number of features, this is required by LayerNorm
            """
        super(Encoder, self).__init__()
        self.resnet_bands = resnet_bands  # images input
        self.resnet_build = resnet_build
        self.Mlp = Mlp  # metadata input
        self.fc = nn.Linear(dim, num_outputs,device=args.gpus)  # combines both together
        self.relu = nn.ReLU()
        self.dim = dim

    def forward(self, x):
        features_img, features_b, features_meta = torch.zeros((x['buildings'].shape[0], self.dim)), torch.zeros(
            (x['buildings'].shape[0], self.dim)), torch.zeros((x['buildings'].shape[0], self.dim))
        features_img = self.resnet_bands(x['images'])[1] if 'images' in x else features_img
        features_b = self.resnet_build(x['buildings'])[1] if 'buildings' in x else features_b
        features_meta = self.Mlp(x[args.metadata[0]])[1] if args.metadata[0] in x else features_meta

        # aggergation:
        features = features_img + features_b + features_meta
        print('features shape together :', features.shape)
        return self.fc(self.relu(features))

