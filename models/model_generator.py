"""The transformer code follows the Annotated Transformer implementation.
See https://nlp.seas.harvard.edu/2018/04/03/attention.html"""
from models.RelPE import RelPosEmb2D
#from models.regnet import regnet_y_400mf

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

model_type = dict(#resnet18=PreActResNet18,
                   resnet18=resnet18,
                  #regnet=regnet_y_400mf,
                  resnet34=resnet34,
                  resnet50=resnet50,
                  mlp=mlp,
                  resnext=resnext50_32x4d,
                  vit=vit_B_32,
                  vitL=vit_L_32,
                  vit16=vit_B_16,
                  vit384=vit_B_32_384
                  )


def taylor_softmax_v1(x, dim=1, n=4, use_log=False):
    assert n % 2 == 0 and n > 0
    fn = torch.ones_like(x)
    denor = 1.
    for i in range(1, n + 1):
        denor *= i
        fn = fn + x.pow(i) / denor
    out = fn / fn.sum(dim=dim, keepdims=True)
    if use_log: out = out.log()
    return out


def get_model(model_name, in_channels, pretrained=False, ckpt_path=None):
    model_fn = model_type[model_name]
    if model_name == 'regnet':
       model = model_fn()
    else:
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

        return self.a_2 * ((x - mean) / (std + self.eps)) + self.b_2  # bs, n, d_model


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout=0):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        # x: bs, n, d_model

        return x + self.dropout(sublayer(self.norm(x)))  # x: bs, n, d_model


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below) -query attention(different from key& value "

    def __init__(self, size, self_attn, feed_forward, dropout=0):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size  # d_model or embed_dim

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x))  # bs, n ,d
        return self.sublayer[1](x, self.feed_forward)  # bs, n , d_model


class Layers(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Layers, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        "Pass the input through each layer in turn."
        for layer in self.layers:
            x = layer(x)  # query , uniform query, random query
        return self.norm(x)  # bs , n , d_model


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
        self.resnet_ms = resnet_ms
        self.resnet_build = resnet_build
        self.Mlp = Mlp
        if self.resnet_build and self.resnet_bands:
            self.fc_in_dim = self.resnet_bands.fc.in_features + self.resnet_build.fc.in_features
        else:
            self.fc_in_dim = self.resnet_bands.fc.in_features
        if self.self_attn == 'multihead_early':
            self.fc_in_dim =  64+128+256
        # self.fc_in_dim = 256

        self.dim = self.fc_in_dim
        # self.pre_norm = LayerNorm(self.fc_in_dim)

        self.ff = nn.Sequential(nn.Linear(self.fc_in_dim, self.fc_in_dim // 4), nn.GELU(),
                                nn.Linear(self.fc_in_dim // 4, self.fc_in_dim))
        self.patch = patch
        self.stride = stride
        self.num_patches = int((
                                       args.crop - self.patch) / self.stride) + 1  # TODO it will produce error in loading pretrained models if args crop changed
        if self.self_attn:
            if self.self_attn == 'multihead_space':

                self.PE = GridCellSpatialRelationEncoder(spa_embed_dim=self.fc_in_dim)
            else:
                self.PE = PositionalEncoding2D(self.fc_in_dim)
            if self.self_attn == 'torch':
                self.multi_head_adapt = MultiheadAttention(input_dim=self.fc_in_dim, embed_dim=self.fc_in_dim,
                                                           num_heads=
                                                           1)
            else:
                self.multi_head_adapt = MultiHeadedAttentionAdapt(h=1, d_model=self.fc_in_dim, w=self.self_attn)
            self.layer_adapt = EncoderLayer(size=self.fc_in_dim, self_attn=self.multi_head_adapt,
                                            feed_forward=self.ff)
            self.layers_adapt = Layers(self.layer_adapt, attn_blocks)

        self.fc = nn.Linear(self.fc_in_dim, num_outputs).to(
            args.gpus)  # combines both together
        with torch.no_grad():
            self.init_weights()

    # @torch.no_grad
    def init_weights(self):
        def initial(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    # nn.init.xavier_normal_(m.bias)

        self.apply(initial)
        if self.self_attn == 'multihead' and isinstance(self.PE, Learnt_PE):
            nn.init.trunc_normal_(self.PE.pos_embedding, std=.02)

    # @autocast()
    def forward(self, x):
        # I'm assuming that I have only one input for now
        features = []
        key = list(x.keys())

        if not self.self_attn:
            features.append(self.resnet_bands(x[key[0]])[1])
            # features = torch.cat(features)
            # x_p = img_to_patch_strided(x[key], p=self.patch, s=self.stride)
            # b, num_patches, c, h, w = x_p.shape

            # for i in range(num_patches):
            #   features.append(self.resnet_bands(x_p[:, i, ...].view(-1, c, h, w))[1])
            features = torch.cat(features)
            # features = torch.stack((features), dim=1)
            # features = torch.mean(features, dim=1, keepdim=False)



        else:
            # patching
            print('in attention with patches')

            x_p = img_to_patch_strided(x[key[0]], p=self.patch, s=self.stride)
            # x_p2=img_to_patch_strided(x['buildings'], p=120,s=100)

            print('patches shape :', x_p.shape)
            b, num_patches, c, h, w = x_p.shape
            # feature extracting
            if self.self_attn == 'multihead_early':
                for p in range(num_patches):
                    _,_,layer3,layer2,layer1=self.resnet_bands(x_p[:, p, ...].view(-1, c, h, w))
                    output=torch.cat((layer3,layer2,layer1),dim=-1)
                    features.append(output)
                    #features.append(self.resnet_bands(x_p[:, p, ...].view(-1, c, h, w))[2])
            else:
                for p in range(num_patches):
                    features.append(self.resnet_bands(x_p[:, p, ...].view(-1, c, h, w))[1])

            features = torch.stack(features, dim=1)
            if self.resnet_build:
                features2 = []initialized
length:  5938
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
<ipython-input-5-09dd68772429> in <module>()
     34     print("length: ", len(df))
     35
---> 36     basemap = ee.ImageCollection("USDA/NAIP/DOQQ").select(['R', 'G', 'B']).filter(ee.Filter.date('2016-01-01', '2018-06-01')).visualize(min=[0.0, 0.0, 0.0], max=[255.0, 255.0, 255.0])
     37
     38     pool = mp.Pool()

AttributeError: 'ImageCollection' object has no attribute 'visualize'
                x_p2 = img_to_patch_strided(x[key[1]], p=self.patch, s=self.stride)
                print('patches for ms shape :', x_p2.shape)
                b, num_patches2, c2, h2, w2 = x_p2.shape
                # feature extracting

                for p in range(num_patches2):
                    features2.append(self.resnet_build(x_p2[:, p, ...].view(-1, c2, h2, w2))[1])

                features2 = torch.stack((features2), dim=1)
                features = torch.cat((features, features2), dim=-1)

            assert tuple(features.shape) == (
                b, num_patches, self.fc_in_dim), 'shape of features after resnet is not as expected'
            if self.self_attn !='global_pool':
                if self.self_attn !='multihead_relative':
                    # Positional encoder
                    features = rearrange(features, 'b (p1 p2) d -> b p1 p2 d', p1=int(num_patches ** 0.5),
                                         p2=int(num_patches ** 0.5))

                    features = self.PE(features)

                    assert tuple(features.shape) == (b, int(num_patches ** 0.5), int(num_patches ** 0.5),
                                                     self.fc_in_dim), 'positional encoding shape is not as expected'
                    features = rearrange(features, 'b p1 p2 d -> b (p1 p2) d', p1=int(num_patches ** 0.5),
                                         p2=int(num_patches ** 0.5))
                    assert tuple(features.shape) == (
                        b, num_patches, self.fc_in_dim), 'rearrange of PE shape is not as expected'
                # Attention Layers
                features = self.layers_adapt(features)
                assert tuple(features.shape) == (
                    b, num_patches, self.fc_in_dim), 'output of  attention layer is not correct'
                # Aggregation
                #if self.self_attn == 'multihead_space':
                 #   features = features[:, (num_patches - 1) // 2, :].squeeze(1)
                 #   assert tuple(features.shape) == (b, self.fc_in_dim), 'aggeragtion output of features is not as expected'
                #else:
                features = torch.mean(features, dim=1, keepdim=False)
                    # concat:
                    # features = rearrange(features, 'b n d -> b (n d)', d=self.fc_in_dim)
                assert tuple(features.shape) == (b, self.fc_in_dim), 'aggeragtion output of features is not as expected'
            else:
                features = torch.mean(features, dim=1, keepdim=False)
        # return self.fc(self.relu(self.dropout(torch.cat(features))))
        return self.fc(self.relu(self.dropout(features)))


def attention(query, key, value, tmp=1, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query: bs, h,n, embed_dim
    # key: bs, h,n, embed_dim
    # value: bs, h, n,embed_dim

    b, h, n, d = query.shape
    # Normalize:
    # query, key = F.normalize(query, dim=-1), F.normalize(key, dim=-1)
    scores = einsum('b h i d, b h j d -> b h i j', query, key) / math.sqrt(d)

    assert tuple(scores.shape) == (b, h, n, n), 'the shape is not as expected'
    p_attn = F.softmax(scores / tmp, dim=-1)
    # p_attn=taylor_softmax_v1(scores/tmp)
    print('scores ', p_attn.shape)

    out = einsum('b h i j, b h j d -> b h i d', p_attn, value)
    assert tuple(out.shape) == (b, h, n, d), 'shape of attention output is not expected'

    return out, p_attn


def attention_uniform(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query: bs, h,n, embed_dim
    # key: bs, h,n, embed_dim
    # value: bs, h, n,embed_dim

    b, h, n, d = query.shape

    scores_identity = torch.ones((b, h, n, n)).type_as(query)

    p_attn_identity = F.softmax(scores_identity, dim=-1)
    print('uniform scores ', p_attn_identity)

    out_ident = einsum('b h i j, b h j d -> b h i d', p_attn_identity, value)
    assert tuple(out_ident.shape) == (b, h, n, d), 'shape of uniform attention output is not expected'

    return out_ident, p_attn_identity


def attention_center(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query: bs, h,n, embed_dim
    # key: bs, h,n, embed_dim
    # value: bs, h, n,embed_dim

    b, h, n, d = query.shape
    # Sanity check for center patch:
    if (n - 1) % 2 != 0:
        raise NotImplementedError
    else:
        query = query[:, :, (n - 1) // 2, :].unsqueeze(1)  # just take the center patch
        assert tuple(query.shape) == (b, 1, 1, d)
    scores = einsum('b h i d, b h j d -> b h i j', query, key) / math.sqrt(d)

    assert tuple(scores.shape) == (b, h, 1, n), 'scores shape is not as expected'

    p_attn = F.softmax(scores, dim=-1)
    print('scores ', p_attn.shape)

    out = einsum('b h i j, b h j d -> b h j d', p_attn, value)
    assert tuple(out.shape) == (b, h, n, d), 'shape of attention output is not expected'

    return out, p_attn


class MultiHeadedAttentionAdapt(nn.Module):
    '''
    attention if just central patch query is used
    '''

    def __init__(self, h, d_model, dropout=0.1, w='multihead'):
        "Take in model size and number of heads."
        super(MultiHeadedAttentionAdapt, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        print('in multiheadedAttention')
        self.d_k = d_model // h

        self.h = h
        self.linears = nn.Linear(d_model, 3 * d_model)
        self.proj_o = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.w = w
        if self.w=='multihead_relative':
            self.PE=RelPosEmb2D((3,3),self.d_k)
        self._reset_parameters()

        # @torch.no_grad

    def _reset_parameters(self):
        def initial(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(
                    m.weight)  # _trunc_normal(m.weight, std=0.02)  # from .initialization import _trunc_normal
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    # nn.init.xavier_normal_(m.bias)

        self.apply(initial)

    def forward(self, x):
        nbatches, nPatches = x.size(0), x.size(1)

        # assert tuple(query.shape) == (nbatches, 1, self.d_k)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        qkv = self.linears(x)
        qkv = qkv.view(nbatches, -1, self.h, 3 * self.d_k)
        qkv = rearrange(qkv, 'b n h d -> b h n d', h=self.h)
        query, key, value = qkv.chunk(3, dim=-1)
        # query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        #                     for l, x in zip(self.linears, (query, key, value))]
        assert tuple(query.shape) == (nbatches, self.h, nPatches, self.d_k)
        assert tuple(key.shape) == (nbatches, self.h, nPatches, self.d_k)
        assert tuple(value.shape) == (nbatches, self.h, nPatches, self.d_k)

        # 2) Apply attention on all the projected vectors in batch.
        if self.w == 'multihead_uniform':
            x, self.attn = attention_uniform(query, key, value,
                                             )
        elif self.w == 'multihead' or self.w == 'multihead_early':
            print('here')
            x, self.attn = attention(query, key, value)
        elif self.w == 'multihead_space':
            x, self.attn = attention_center(query, key, value)

        elif self.w =='multihead_relative':
            qr=self.PE(query)
            assert tuple(qr.shape) == (nbatches, self.h, nPatches, nPatches)
            scores = einsum('b h i d, b h j d -> b h i j', query, key) / math.sqrt(self.d_k)

            assert tuple(scores.shape) == (nbatches, self.h, nPatches, nPatches), 'the shape is not as expected'
            self.attn = F.softmax(scores+qr, dim=-1)
            # p_attn=taylor_softmax_v1(scores/tmp)
            print('scores ', self.attn.shape)

            x = einsum('b h i j, b h j d -> b h i d', self.attn, value)
            assert tuple(x.shape) == (nbatches, self.h, nPatches, self.d_k), 'shape of attention output is not expected'




        else:
            raise NotImplementedError

        # 3) "concat heads "
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.h)

        assert tuple(x.shape) == (nbatches, nPatches, self.d_k * self.h)

        return self.dropout(self.proj_o(x))  # bs , n , d_model


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.attn = None

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):

        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        self.attn = attention

        if return_attention:
            return o, attention
        else:
            return o


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
    patches_shape = patches.shape
    # print('strided patches size :', patches.shape)  # should be b x c x num_patchesx num_patches x 100 x 100
    num_patches1, num_patches2 = patches.shape[2], patches.shape[3]
    # num_patches=((H-100)/s +1) **2

    patches = rearrange(patches, 'b c p1 p2 h w -> b (p1 p2) c h w ', p1=num_patches1, p2=num_patches2, h=p, w=p)
    # print('strided patch after rearrange ', patches.shape)
    # Sanity check for equality (reshape back)
    if s==p:
        patches_orig = patches.view(patches_shape)
        output_h, output_w = patches_shape[2] * patches_shape[4], patches_shape[3] * patches_shape[5]
        patches_orig = rearrange(patches_orig, 'b c p1 p2 h w -> b c (p1 h) (p2 w)', p1=num_patches1, p2=num_patches2, h=p,
                                 w=p)
        assert tuple(patches_orig.shape) == (
        img.shape[0], img.shape[1], output_h, output_w), 'patches original shape is not as expected'
        #TODO What if I'm using padding?
        assert torch.all(patches_orig.eq(img[:,:,:output_h, :output_w])), 'orginal tensor isnot as same as patched one'
    '''
    else:
      
        # REASSEMBLE THE IMAGE USING FOLD
        patches_orig = patches.contiguous().view(img.shape[0], img.shape[1], -1, p * p)
        patches_orig = patches_orig.permute(0, 1, 3, 2)
        patches_orig = patches_orig.contiguous().view(img.shape[0],img.shape[1], p * p, -1)
        patches_orig = F.fold(patches_orig, output_size=(img.shape[-2], img.shape[-1]), kernel_size=p, stride=s)
        patches_orig = patches_orig.squeeze()
        assert torch.all(
            patches_orig.eq(img), 'orginal tensor isnot as same as patched one'
    '''
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
        # print('position_embedding of the first channel: ', self.pos_cache[0, :, :, 0], 'second channel: ',
        #     self.pos_cache[0, :, :, 1])
        return emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1) + tensor


class PositionalEncoding1D(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


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
        self.dropout = nn.Dropout(p=0)
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
    print('scores ', p_attn)
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


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        print('in multiheadedAttention')
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 3)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn, = attention(query, key, value
                                  )

        # 3) "Concat" using a view and apply a final linear.(done here already in the attention function)
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.h)

        # x = x.transpose(1, 2).contiguous().view(
        #   nbatches, -1, self.h * self.d_k)  # bs , n , d_model
        # x=x.reshape(b,n,h*d)
        # output = self.linears[-1](x)
        return x  # bs , n , d_model


"""
  elif self.self_attn == 'multihead':
                print(' inside attention that use positional encoding')
                features = rearrange(features, 'b (p1 p2) d -> b p1 p2 d', p1=int(num_patches ** 0.5),
                                     p2=int(num_patches ** 0.5))

                features = self.PE(features)
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
                # features = torch.mean(features, dim=1, keepdim=False)
                # concat:
                features = rearrange(features, 'b n d -> b (n d)', d=self.fc_in_dim)
                assert tuple(features.shape) == (b, self.dim), 'aggeragtion output of features is not as expected'
            else:
                features = features.squeeze(1)     


def attention(query, key, value, tmp=.01, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query: bs, h,n, embed_dim
    # key: bs, h,n, embed_dim
    # value: bs, h, n,embed_dim

    b, h, n, d = query.shape
    #Normalize:
    query,key=F.normalize(query,dim=-1),F.normalize(key,dim=-1)
    scores = einsum('b h i d, b h j d -> b h i j', query, key) / math.sqrt(d)

    assert tuple(scores.shape) == (b, h, n, n), 'the shape is not as expected'
    p_attn = F.softmax(scores / tmp, dim=-2)
    #p_attn=taylor_softmax_v1(scores/tmp)
    print('scores ', p_attn.shape)

    out = einsum('b h i j, b h i d -> b h i d', p_attn, value)
    assert tuple(out.shape) == (b, h, n, d), 'shape of attention output is not expected'

    return out, p_attn   
"""
