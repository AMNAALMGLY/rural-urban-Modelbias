"""The transformer code follows the Annotated Transformer implementation.
See https://nlp.seas.harvard.edu/2018/04/03/attention.html"""

"""position embedding from https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings
/positional_encodings.py """

import copy
import math

import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange
from torch.cuda.amp import autocast

from configs import args
from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50
# from models.vit import vit_small_patch32_224
from models.pytorch_pretrained_vit.model import vit_B_32, vit_B_16, vit_L_32, vit_B_32_384
from models.resnet import resnet18, resnet34, resnet50, mlp, resnext50_32x4d
from models.spaceEncoder import GridCellSpatialRelationEncoder
from utils.utils import load_from_checkpoint
import torch.nn.functional as F

model_type = dict(resnet18=PreActResNet18,
                  resnet34=resnet34,
                  resnet50=resnet50,
                  mlp=mlp,
                  resnext=resnext50_32x4d,
                  vit=vit_B_32,
                  vitL=vit_L_32,
                  vit16=vit_B_16,
                  vit384=vit_B_32_384
                  )


def get_model(model_name, in_channels, pretrained=False, ckpt_path=None):
    model_fn = model_type[model_name]
    # if model_name == 'vit':
    #   model = model_fn()
    # else:
    model = model_fn(in_channels, pretrained)
    if ckpt_path:
        model = load_from_checkpoint(ckpt_path, model)
    return model


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        # x: bs, n, d_model

        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2  # bs, n, d_model


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # x: bs, n, d_model

        return x + self.dropout(sublayer(self.norm(x)))  # x: bs, n, d_model


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size  # d_model or embed_dim

    def forward(self, q, k, v):
        x = self.sublayer[0](q, lambda q: self.self_attn(q, k, v)[0])  # bs, n ,d
        y = self.sublayer[0](q, lambda q: self.self_attn(q, k, v)[1])  # bs, n ,d
        z = self.sublayer[0](q, lambda q: self.self_attn(q, k, v)[2])  # bs, n ,d

        return self.sublayer[1](x, self.feed_forward), self.sublayer[1](y, self.feed_forward), self.sublayer[1](z,
                                                                                                                self.feed_forward)  # bs, n , d_model


class Layers(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Layers, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, q, k, v):
        "Pass the input through each layer in turn."
        for layer in self.layers:
            x, y, z = layer(q, k, v)
        return self.norm(x), self.norm(y), self.norm(z)  # bs , n , d_model


class Encoder(nn.Module):
    def __init__(self, resnet_build=None, resnet_bands=None, resnet_ms=None, Mlp=None, self_attn=None, attn_blocks=6,
                 patch=100, stride=50, dim=512,
                 num_outputs=1,
                 model_dict=None):
        # TODO add resnet_NL and resnet_Ms
        # TODO add multiple mlps for metadata
        """
            Args:

                resnet_bands (nn.Module): cnn for the mutlispectral input if used
                resnet_build (nn.Module): cnn for the building input if used
                MLp (int): MLP for metadat if used
                self_attn(str) : either None, multihead space , multihead , multihead uniform or multihead random depending on the experiment
                patch=patch size
                stride: step
                num_output:output of the fc layer
            """
        super(Encoder, self).__init__()
        # self.models = nn.ModuleDict({key:value for key, value in model_dict.items()})
        # print('Module dict ',self.models)
        # self.fc_in_dim = dim * len(list(model_dict.values()))  # concat dimension depends on how many models I have

        self.relu = nn.ELU()
        self.dropout = nn.Dropout(p=0.1)
        self.self_attn = self_attn
        # MultiHeadedAttention(h=1,d_model=512)

        self.resnet_bands = resnet_bands
        self.fc_in_dim = self.resnet_bands.fc.in_features
        self.fc = nn.Linear(self.fc_in_dim, num_outputs, device=args.gpus)  # combines both together
        self.dim = self.fc_in_dim

        self.resnet_ms = resnet_ms
        self.resnet_build = resnet_build
        self.Mlp = Mlp

        self.ff = nn.Sequential(nn.Linear(self.fc_in_dim, self.fc_in_dim // 2), nn.GELU(),
                                nn.Linear(self.fc_in_dim // 2, self.fc_in_dim))
        self.patch = patch
        self.stride = stride
        self.num_patches = int((355 - self.patch) / self.stride) + 1

        if self_attn == 'multihead_space':

            self.spaceE = GridCellSpatialRelationEncoder(spa_embed_dim=self.fc_in_dim)
            self.multi_head_adapt = MultiHeadedAttention_adapt(h=1, d_model=self.fc_in_dim)
            self.layer_adapt = EncoderLayer(size=self.fc_in_dim, self_attn=self.multi_head_adapt, feed_forward=self.ff)
            self.layers_adapt = Layers(self.layer_adapt, attn_blocks)

        elif self_attn:
            # self.positionalE = PositionalEncoding2D(self.fc_in_dim)
            self.positionalE = Learnt_PE(self.num_patches, self.fc_in_dim)
            self.multi_head = MultiHeadedAttention(h=1, d_model=self.fc_in_dim)
            self.layer = EncoderLayer(size=self.fc_in_dim, self_attn=self.multi_head, feed_forward=self.ff)
            self.layers = Layers(self.layer, attn_blocks)

    # @autocast()
    def forward(self, x):
        features = []
        key = list(x.keys())[0]

        if not self.self_attn:
            features.append(self.resnet_bands(x[key])[1])
            # features = torch.cat(features)
            # x_p = img_to_patch_strided(x[key], p=self.patch, s=self.stride)
            # b, num_patches, c, h, w = x_p.shape

            # for i in range(num_patches):
            #   features.append(self.resnet_bands(x_p[:, i, ...].view(-1, c, h, w))[1])
            features = torch.cat(features)
            # features = torch.stack((features), dim=1)
            # features = torch.mean(features, dim=1, keepdim=False)



        else:
            print('in attention with patches')
            x_p = img_to_patch_strided(x[key], p=self.patch, s=self.stride)
            # x_p2=img_to_patch_strided(x['buildings'], p=120,s=100)

            print('patches shape :', x_p.shape)
            b, num_patches, c, h, w = x_p.shape

            for p in range(num_patches):
                features.append(self.resnet_bands(x_p[:, p, ...].view(-1, c, h, w))[1])
            # features2.append(self.resnet_ms(x_p2[:, p, ...].view(-1, c2, h2, w2))[1])
            features = torch.stack((features), dim=1)

            assert tuple(features.shape) == (
                b, num_patches, self.fc_in_dim), 'shape of features after resnet is not as expected'

            if self.self_attn == 'multihead_space':
                print(' inside space  attention')
                # features = rearrange(features, 'b (p1 p2) d -> b p1 p2 d', p1=int(num_patches ** 0.5),
                #                    p2=int(num_patches ** 0.5))

                features = self.spaceE(features)

                # assert tuple(features.shape) == (b, int(num_patches ** 0.5), int(num_patches ** 0.5),
                #                                self.fc_in_dim), 'positional encoding shape is not as expected'
                # features = rearrange(features, 'b p1 p2 d -> b (p1 p2) d', p1=int(num_patches ** 0.5),
                #                    p2=int(num_patches ** 0.5))
                assert tuple(features.shape) == (
                    b, num_patches, self.fc_in_dim), 'rearrange of PE shape is not as expected'

                query = features[:, (num_patches - 1) // 2, :].unsqueeze(1)  # just take the center patch
                print(query.shape)
                features, _, _ = self.layers_adapt(query, features, features)
                assert tuple(features.shape) == (
                    b, 1, self.fc_in_dim), 'output of space attention layer is not correct'


            elif self.self_attn == 'multihead':
                print(' inside attention that use positional encoding')
                features = rearrange(features, 'b (p1 p2) d -> b p1 p2 d', p1=int(num_patches ** 0.5),
                                     p2=int(num_patches ** 0.5))

                features = self.positionalE(features)
                assert tuple(features.shape) == (b, int(num_patches ** 0.5), int(num_patches ** 0.5),
                                                 self.fc_in_dim), 'positional encoding shape is not as expected'
                features = rearrange(features, 'b p1 p2 d -> b (p1 p2) d', p1=int(num_patches ** 0.5),
                                     p2=int(num_patches ** 0.5))
                assert tuple(features.shape) == (
                    b, num_patches, self.fc_in_dim), 'rearrange of PE shape is not as expected'
                features, _, _ = self.layers(features, features, features)
                assert tuple(features.shape) == (b, num_patches, self.fc_in_dim), 'output of positional attention ' \
                                                                                  'layer is not correct '


            elif self.self_attn == 'multihead_uniform':
                print(' inside uniform attention')

                _, features, _ = self.layers(features, features, features)
                assert tuple(features.shape) == (b, num_patches, self.fc_in_dim), 'output of uniform attention ' \
                                                                                  'layer is not correct '

            elif self.self_attn == 'multihead_random':
                print(' inside random attention')

                _, _, features = self.layers(features, features, features)
                assert tuple(features.shape) == (b, num_patches, self.fc_in_dim), 'output of random attention ' \
                                                                                  'layer is not correct '

            if features.size(
                    1) > 1:
                features = torch.mean(features, dim=1, keepdim=False)
            else:
                features = features.squeeze(1)

        # return self.fc(self.relu(self.dropout(torch.cat(features))))
        return self.fc(self.relu(self.dropout(features)))


def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query: bs, h,n, embed_dim
    # key: bs, h,n, embed_dim
    # value: bs, h, n,embed_dim

    b, h, n, d = query.shape
    scores = einsum('b h i d, b h j d -> b h i j', query, key) / math.sqrt(d)
    print('scores shape', scores.shape)
    assert tuple(scores.shape) == (b, h, n, n), 'the shape is not as expected'
    p_attn = F.softmax(scores, dim=-1)

    scores_identity = torch.ones_like(scores)
    scores_identity = scores_identity.type_as(scores)
    p_attn_identity = F.softmax(scores_identity, dim=-1)

    scores_random = torch.randn_like(scores)
    scores_random = scores_random.type_as(scores)
    p_attn_random = F.softmax(scores_random, dim=-1)

    out = einsum('b h i j, b h i d -> b h i d', p_attn, value)
    assert tuple(out.shape) == (b, h, n, d), 'shape of attention output is not expected'
    out_ident = einsum('b h i j, b h i d -> b h i d', p_attn_identity, value)
    out_random = einsum('b h i j, b h i d -> b h i d', p_attn_random, value)

    return out, out_ident, out_random, p_attn, p_attn_identity, p_attn_random


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        print('in multiheadedAttention')
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn, self.ident_attn, self.rand_attn = None, None, None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, y, z, self.attn, self.ident_attn, self.rand_attn = attention(query, key, value,
                                                                        )

        # 3) "Concat" using a view and apply a final linear.(done here already in the attention function)
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.h)
        y = rearrange(y, 'b h n d -> b n (h d)', h=self.h)
        z = rearrange(z, 'b h n d -> b n (h d)', h=self.h)

        # x = x.transpose(1, 2).contiguous().view(
        #   nbatches, -1, self.h * self.d_k)  # bs , n , d_model
        # x=x.reshape(b,n,h*d)

        return self.linears[-1](x), y, z  # bs , n , d_model


def attention_adapt(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention if central patch is used as query"
    # query: bs, h,1, embed_dim
    # key: bs, h,n, embed_dim
    # value: bs, h, n,embed_dim

    b, h, n, d = key.shape
    scores = einsum('b h i d, b h j d -> b h i j', query, key) / math.sqrt(d)
    assert scores.shape == (b, h, 1, n), 'the shape is not as expected'
    # tmp = 0.0001
    p_attn = F.softmax(scores, dim=-1)

    scores_identity = torch.ones_like(scores)
    scores_identity = scores_identity.type_as(scores)
    p_attn_identity = F.softmax(scores_identity, dim=-1)

    scores_random = torch.randn_like(scores)
    scores_random = scores_random.type_as(scores)
    p_attn_random = F.softmax(scores_random, dim=-1)

    out = einsum('b h i j, b h j d -> b h i d', p_attn, value)
    assert tuple(out.shape) == (b, h, 1, d), 'shape of out attention isnot as expected'
    out_ident = einsum('b h i j, b h j d -> b h i d', p_attn_identity, value)
    out_random = einsum('b h i j, b h j d -> b h i d', p_attn_random, value)
    # print('output attention ', out.shape)

    return out, out_ident, out_random, p_attn, p_attn_identity, p_attn_random


class MultiHeadedAttention_adapt(nn.Module):
    '''
    attention if just central patch query is used
    '''

    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention_adapt, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        print('in multiheadedAttention')
        self.d_k = d_model // h

        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn, self.ident_attn, self.rand_attn = None, None, None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        nPatches = key.size(1)

        assert tuple(query.shape) == (nbatches, 1, self.d_k)
        # 1) Do all the linear projections in batch from d_model => h x d_k

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        assert tuple(query.shape) == (nbatches, self.h, 1, self.d_k)
        assert tuple(key.shape) == (nbatches, self.h, nPatches, self.d_k)
        assert tuple(value.shape) == (nbatches, self.h, nPatches, self.d_k)

        # 2) Apply attention on all the projected vectors in batch.
        x, y, z, self.attn, self.ident_attn, self.rand_attn = attention_adapt(query, key, value,
                                                                              )

        # 3) "concat heads "
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.h, n=1)
        assert tuple(x.shape) == (nbatches, 1, (self.d_k) * self.h)

        y = rearrange(y, 'b h n d -> b n (h d)', h=self.h)
        z = rearrange(z, 'b h n d -> b n (h d)', h=self.h)

        return self.linears[-1](x), y, z  # bs , n , d_model


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def img_to_patch(img, p):
    # p is patch_size  # P in maths

    x_p = rearrange(img, 'b c (h p1) (w p2) -> b (h w) c p1 p2 ', p1=p, p2=p)
    return x_p


def img_to_patch_strided(img, p=100, s=50, padding=False):
    # p is patch size
    # s is the strid
    # img shape is b c h w
    # calculate padding
    pad0_left = (img.size(2) // s * s + p) - img.size(2)
    pad1_left = (img.size(3) // s * s + p) - img.size(3)

    # Calculate symmetric padding
    pad0_right = pad0_left // 2 if pad0_left % 2 == 0 else pad0_left // 2 + 1
    pad1_right = pad1_left // 2 if pad1_left % 2 == 0 else pad1_left // 2 + 1

    pad0_left = pad0_left // 2
    pad1_left = pad1_left // 2
    if padding:
        img = torch.nn.functional.pad(img, (pad1_left, pad1_right, pad0_left, pad0_right))
        #
        print('shape after padding', img.shape)

    patches = img.unfold(2, p, s).unfold(3, p, s)
    print('strided patches size :', patches.shape)  # should be b x c x num_patchesx num_patches x 100 x 100
    num_patches1, num_patches2 = patches.shape[2], patches.shape[3]
    # num_patches=((H-100)/s +1) **2

    patches = rearrange(patches, 'b c p1 p2 h w -> b (p1 p2) c h w ', p1=num_patches1, p2=num_patches2, h=p, w=p)
    print('strided patch after rearrange ', patches.shape)
    return patches


def intersample_attention(query, key, value):
    "Calculate the intersample of a given query batch"
    # x , bs , n , d

    b, h, n, d = query.shape
    # print(query.shape,key.shape, value.shape )
    query, key, value = query.reshape(1, h, b, n * d), \
                        key.reshape(1, h, b, n * d), \
                        value.reshape(1, h, b, n * d)
    # query, key, value = query.reshape(h,n, b, d), \
    #                    key.reshape(h,n, b, d), \
    #                    value.reshape(h,n, b, d)

    output, scores = attention(query, key, value)  # 1 , h,b, n*d
    output = output.squeeze(0)  # h, b,n*d
    output = output.view(b, h, n, d)  # b,h,n,d
    # output = output.squeeze(0)  # b, n*d
    # output = output.reshape(b, n, d)  # b,n,d

    return output, scores


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels

        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))  # shape:[channels/2]
        self.register_buffer("inv_freq", inv_freq)
        self.pos_cache = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(float(x), device=tensor.device).type(self.inv_freq.type())  # shape: [x]
        pos_y = torch.arange(float(y), device=tensor.device).type(self.inv_freq.type())  # shape: [y]
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)  # shape : [x,channels/2]
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)  # shape:  [y,channels/2]
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)  # [x,1,channels]
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)  # [y,channels]

        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels: 2 * self.channels] = emb_y
        self.pos_cache = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        print('position_embedding of the first channel: ', self.pos_cache[0, :, :, 0], 'second channel: ',
              self.pos_cache[0, :, :, 1])
        return emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1) + tensor


class geoAttention(nn.Module):
    def __init__(self, dim=512,
                 num_outputs=1,
                 ):
        # TODO add resnet_NL and resnet_Ms
        # TODO add multiple mlps for metadata
        """
            Args:

                resnet_bands (nn.Module): Encoder layer of the self attention
                resnet_build (nn.Module): Encoder layer of intersample attention
                MLp (int): Number of features, this is required by LayerNorm
            """
        super(geoAttention, self).__init__()

        self.fc_in_dim = dim
        self.fc = nn.Linear(dim, num_outputs, device=args.gpus)  # combines both together

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(dim * 2, dim)
        # MultiHeadedAttention(h=1,d_model=512)
        self.dim = dim

        self.multi_head = MultiHeadedAttention(h=1, d_model=dim * 2)

        # nn.MultiheadAttention(self.dim, 1)

    def forward(self, x):
        # features = torch.stack((x),dim=1)
        b, d = x.shape
        features = x.reshape(b, 1, d)

        print('features_concat_shape', features.shape)

        print('in attention')

        self.multi_head.to(args.gpus)
        attn, _ = self.multi_head(features, features, features)

        print('attention shape', attn.shape)
        features = features + attn  # residual connection
        print('features shape', features.shape)
        features = torch.sum(features, dim=1, keepdim=False)
        # features = features.view(-1, self.fc_in_dim)
        print('features shape after sum', features.shape)

        print('shape of fc', self.relu(self.dropout(self.linear(features))).shape)
        return self.fc(self.relu(self.dropout(self.linear(features))))


class Learnt_PE(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def __init__(self, seq_len, dim):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, seq_len, dim))

    def forward(self, x):
        """Input has shape `(batch_size, seq_len,seq_len, emb_dim)`"""
        print('learnt_embedding', self.pos_embedding[0, :, :, 0], 'second_channel', self.pos_embedding[0, :, :, 1])
        return x + self.pos_embedding
