import copy
import math

import torch
from torch import nn

from configs import args
from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50
from models.resnet import resnet18, resnet34, resnet50, mlp
from utils.utils import load_from_checkpoint
import torch.nn.functional as F
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
    def __init__(self, resnet_bands=None, resnet_build=None, Mlp=None, self_attn=None,dim=512, num_outputs=1):
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
        self.dropout=nn.Dropout(p=0.1)
        self.self_attn=self_attn
        self.dim = dim

    def forward(self, x):
        features_img, features_b, features_meta = torch.zeros((x['buildings'].shape[0], self.dim),device=args.gpus)\
            , torch.zeros(
            (x['buildings'].shape[0], self.dim),device=args.gpus), torch.zeros((x['buildings'].shape[0], self.dim),device=args.gpus)
        features_img = self.resnet_bands(x['images'])[1] if 'images' in x else features_img
        features_b = self.resnet_build(x['buildings'])[1] if 'buildings' in x else features_b
        features_meta = self.Mlp(x[args.metadata[0]])[1] if args.metadata[0] in x else features_meta

        # aggergation:
        features = features_img + features_b + features_meta
        print('features shape together :', features.shape)
        attn=self.dropout(self.self_attn(features,features,features))
        print('attention shape',attn.shape)
        features=features+attn
        return self.fc(self.relu(features))


def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query: bs, n, embed_dim
    # key: bs, n, embed_dim
    # value: bs, n, embed_dim
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)  # bs , n , n
    output = torch.matmul(p_attn, value)  # bs, n , embed_dim
    return output, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)  # bs , n , d_model
        return self.linears[-1](x)  # bs , n , d_model

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])