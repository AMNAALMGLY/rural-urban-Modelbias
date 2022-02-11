import copy
import math

import numpy as np
import torch
from torch import nn, einsum
from einops import rearrange
from configs import args
from models.preact_resnet import PreActResNet18, PreActResNet34, PreActResNet50
from models.resnet import resnet18, resnet34, resnet50, mlp
from utils.utils import load_from_checkpoint
import torch.nn.functional as F

model_type = dict(resnet18=PreActResNet18,
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


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(float(x), device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(float(y), device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        print(emb_x.requires_grad)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels: 2 * self.channels] = emb_y
        print(emb.requires_grad)
        return emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1) + tensor


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size  # d_model or embed_dim

    def forward(self, x):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x))  # bs, n ,d

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
            x = layer(x)
        return self.norm(x)  # bs , n , d_model


class Encoder(nn.Module):
    def __init__(self, resnet_build=None, resnet_bands=None, resnet_ms=None, Mlp=None, self_attn=None,dim=512,
                 num_outputs=1,
                 model_dict=None, freeze_encoder=False):
        # TODO add resnet_NL and resnet_Ms
        # TODO add multiple mlps for metadata
        """
            Args:

                resnet_bands (nn.Module): Encoder layer of the self attention
                resnet_build (nn.Module): Encoder layer of intersample attention
                MLp (int): Number of features, this is required by LayerNorm
            """
        super(Encoder, self).__init__()
        # self.models = nn.ModuleDict({key:value for key, value in model_dict.items()})
        # print('Module dict ',self.models)
        # self.fc_in_dim = dim * len(list(model_dict.values()))  # concat dimension depends on how many models I have

        self.relu = nn.ReLU()
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

        self.positionalE = PositionalEncoding2D(channels=120*120*1)
        #self.pe=torch.empty((args.batch_size,4,self.dim),requires_grad=True)
        self.multi_head = MultiHeadedAttention(h=1, d_model=self.fc_in_dim)
        self.ff = nn.Linear(self.fc_in_dim, self.fc_in_dim)
        self.layer = EncoderLayer(size=self.fc_in_dim, self_attn=self.multi_head, feed_forward=self.ff)
        self.layers = Layers(self.layer, 6)
        # nn.MultiheadAttention(self.dim, 1)

    def forward(self, x):
        features = []
        # for  key in  x.keys():
        # print(f'appending {model_name} features', type(model),x[key].requires_grad)
        # self.models[model_name].to(args.gpus)
        # feature = torch.tensor(self.models[model_name](x[key])[1], device=args.gpus)
        # features.append(feature)
        #features.append(self.resnet_bands(x['images'])[1])
        #features.append(self.resnet_ms(x['ms'])[1])

        # patches Experiments
        #print('image shape',x['images'].shape)
        #just for the NL+b experiment
        #x['buildings']=torch.cat((x['buildings'],x['images']),dim=1)
       # print('images ', x['images'])
        x_p = img_to_patch_strided(x['images'], p=120)

        print('patches shape :', x_p.shape)
        b, num_patches, c, h, w = x_p.shape


        '''
        for p in range(num_patches):

            features.append(self.resnet_bands(x_p[:, p, ...].view(-1, c, h, w))[1])
        features = torch.stack((features), dim=1)
        '''
        #Vectorization

        features=torch.empty((b,num_patches,self.fc_in_dim))
        features=features.type_as(x_p)
        #features[:, range(num_patches), :]
        #res_input=torch.empty((b,c,h,w)).type_as(x_p)


        features[:,torch.arange(num_patches),:]=self.resnet_bands(x_p[:, 0, ...])[1]

        # features.append(self.resnet_build(x['buildings'])[1])

        # if self.Mlp:
        #    assert len(list(x[args.metadata[0]].shape)) == 2, 'the number of dimension should be 2'
        #    number_of_fts = x[args.metadata[0]].shape[-1]
        #    assert 2 >= number_of_fts >= 1, 'number of features should be at least one'
        #    features.append(self.Mlp(torch.cat([x[args.metadata[0]], x[args.metadata[1]]], dim=-1))[1])
        #

        assert tuple(features.shape) == (b, num_patches, self.fc_in_dim), 'shape is not as expected'

        print('features_concat_shape', features.shape)

        features=rearrange(features, 'b (p1 p2) d -> b p1 p2 d', p1=int(num_patches ** 0.5),
                             p2=int(num_patches ** 0.5))
        
        features=self.positionalE(features)
        assert tuple(features.shape) == (b, int(num_patches**0.5),int(num_patches**0.5), self.fc_in_dim), 'positional encoding shape is not as expected'
        features= rearrange(features, 'b p1 p2 d -> b (p1 p2) d', p1=int(num_patches ** 0.5),
                             p2=int(num_patches ** 0.5))
        assert tuple(features.shape) == (b, num_patches, self.fc_in_dim), 'rearrange of PE shape is not as expected'




        if self.self_attn:
            print('in attention')

            if self.self_attn == 'vanilla':
                attn, _ = attention(features, features, features)  # bxnxd
            elif self.self_attn == 'intersample':

                attn, _ = intersample_attention(features, features, features)  # bxnxd
            elif self.self_attn == 'multihead':
                print(' inside multi head attention')

                features = self.layers(features)

                # self.multi_head.to(args.gpus)
                # attn = self.multi_head(features, features, features)
            features = torch.max(features, dim=1, keepdim=False)[0]
            # print('attention shape', attn.shape)


        #return self.fc(self.relu(self.dropout(torch.cat(features))))
        return self.fc(self.relu(self.dropout(features)))

    """
        features_img, features_b, features_meta = torch.zeros((x['buildings'].shape[0], self.dim), device=args.gpus) \
            , torch.zeros(
            (x['buildings'].shape[0], self.dim), device=args.gpus), torch.zeros((x['buildings'].shape[0], self.dim),
                                                                                device=args.gpus)
        features_img = self.resnet_bands(x['images'])[1] if 'images' in x else features_img
        features_b = self.resnet_build(x['buildings'])[1] if 'buildings' in x else features_b

        features_meta = self.Mlp(x[args.metadata[0]])[1] if args.metadata else features_meta
        
        # aggergation:
        # features = features_img + features_b + features_meta
        # features=torch.mean(features,dim=1,keepdim=False)
        # print('fc features',features.shape)
        # TODO concatination when some outputs are none
        if self.self_attn:
            batch = features_img.shape[0]
            print('in attention')
            features_img.unsqueeze_(-1)
            features_b.unsqueeze_(-1)
            features_meta.unsqueeze_(-1)  # bxdx1
            features_concat = torch.cat([features_img, features_b], dim=-1)  # bxdx3
            features_concat = features_concat.transpose(-2, -1)  # bx3xd
            # print('features shape together :', features_concat.shape)
            #attn=self.dropout(self.self_attn(features_concat,features_concat,features_concat))
            attn, _ = intersample_attention(features_concat, features_concat, features_concat)  # bx3xd
            print('attention shape', attn.shape)
            features = features_concat + attn

            return self.fc(self.relu(features.reshape(batch, -1)))
        else:
            features_concat = self.dropout(torch.cat([features_img, features_b, ], dim=-1))
            return self.fc(self.relu(features_concat))
        """


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


def attention(query, key, value, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query: bs, h,n, embed_dim
    # key: bs, h,n, embed_dim
    # value: bs, h, n,embed_dim
    '''
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    print('scores shape', scores.shape)
    p_attn = F.softmax(scores, dim=-1)
    print('softmax', p_attn.shape)
    if dropout is not None:
        p_attn = dropout(p_attn)  # bs , n , n
    output = torch.matmul(p_attn, value)  # bs, n , embed_dim
    return output, p_attn
    '''
    b, h, n, d = query.shape
    scores = einsum('b h i d, b h j d -> b h i j', query, key) / math.sqrt(d)
    print('scores shape', scores.shape)
    assert scores.shape == (b, h, n, n), 'the shape is not as expected'
    p_attn = F.softmax(scores, dim=-1)
    out = einsum('b h i j, b h i d -> b h i d', p_attn, value)
    print('output before rearrange ', out.shape)

    return out, p_attn


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
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value,
                                 )

        # 3) "Concat" using a view and apply a final linear.(done here already in the attention function)
        x = rearrange(x, 'b h n d -> b n (h d)', h=self.h)
        print('output of attn ', x.shape)
        # x = x.transpose(1, 2).contiguous().view(
        #   nbatches, -1, self.h * self.d_k)  # bs , n , d_model
        # x=x.reshape(b,n,h*d)
        return self.linears[-1](x)  # bs , n , d_model


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def img_to_patch(img, p):
    # p is patch_size  # P in maths

    x_p = rearrange(img, 'b c (h p1) (w p2) -> b (h w) c p1 p2 ', p1=p, p2=p)
    return x_p
def img_to_patch_strided(img, p=120,s=100,padding=False):
    #p is patch size
    #s is the strid
    #img shape is b c h w
    #calculate padding
    pad0_left = (img.size(2) // s * s + p) - img.size(2)
    pad1_left = (img.size(3) // s * s + p) - img.size(3)


    # Calculate symmetric padding
    pad0_right = pad0_left // 2 if pad0_left % 2 == 0 else pad0_left // 2 + 1
    pad1_right = pad1_left // 2 if pad1_left % 2 == 0 else pad1_left // 2 + 1

    pad0_left = pad0_left // 2
    pad1_left = pad1_left // 2
    if padding:
               img=torch.nn.functional.pad(img,(pad1_left,pad1_right,pad0_left,pad0_right))
   #
               print('shape after padding',img.shape)

    patches=img.unfold(2,p,s).unfold(3,p,s)
    print('strided patches size :', patches.shape)  # should be b x c x num_patchesx num_patches x 100 x 100
    num_patches1,num_patches2=patches.shape[2],patches.shape[3]
    #num_patches=((H-100)/s +1) **2

    patches=rearrange(patches,'b c p1 p2 h w -> b (p1 p2) c h w ',p1=num_patches1,p2=num_patches2,h=p,w=p)
    print('strided patch after rearrange ',patches.shape)
    return patches
