import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np
import math
from einops import rearrange

from configs import args

"""
A Set of position encoder
"""


def _cal_freq_list(freq_init, frequency_num, max_radius, min_radius):
    if freq_init == "random":
        # the frequence we use for each block, alpha in ICLR paper
        # freq_list shape: (frequency_num)
        freq_list = np.random.random(size=[frequency_num]) * max_radius
    elif freq_init == "geometric":
        # freq_list = []
        # for cur_freq in range(frequency_num):
        #     base = 1.0/(np.power(max_radius, cur_freq*1.0/(frequency_num-1)))
        #     freq_list.append(base)

        # freq_list = np.asarray(freq_list)

        log_timescale_increment = (math.log(float(max_radius) / float(min_radius)) /
                                   (frequency_num * 1.0 - 1))

        timescales = min_radius * np.exp(
            np.arange(frequency_num).astype(float) * log_timescale_increment)

        freq_list = 1.0 / timescales

    return freq_list


class GridCellSpatialRelationEncoder(nn.Module):
    """
    Given a list of (deltaX,deltaY), encode them using the position encoding function
    """

    def __init__(self, spa_embed_dim, coord_dim=2, frequency_num=128,
                 max_radius=10000, min_radius=10,
                 freq_init="geometric",
                 ffn=None, device=args.gpus):
        """
        Args:
            spa_embed_dim: the output spatial relation embedding dimention
            coord_dim: the dimention of space, 2D, 3D, or other
            frequency_num: the number of different sinusoidal with different frequencies/wavelengths
            max_radius: the largest context radius this model can handle
        """
        super(GridCellSpatialRelationEncoder, self).__init__()
        self.spa_embed_dim = spa_embed_dim
        self.coord_dim = coord_dim
        self.frequency_num = frequency_num
        self.freq_init = freq_init
        self.max_radius = max_radius
        self.min_radius = min_radius
        # the frequence we use for each block, alpha in ICLR paper
        self.cal_freq_list()
        self.cal_freq_mat()

        self.input_embed_dim = self.cal_input_dim()

        self.ffn = ffn
        self.device = device

    def cal_elementwise_angle(self, coord, cur_freq):
        '''
        Args:
            coord: the deltaX or deltaY
            cur_freq: the frequency
        '''
        return coord / (np.power(self.max_radius, cur_freq * 1.0 / (self.frequency_num - 1)))

    def cal_coord_embed(self, coords_tuple):
        embed = []
        for coord in coords_tuple:
            for cur_freq in range(self.frequency_num):
                embed.append(math.sin(self.cal_elementwise_angle(coord, cur_freq)))
                embed.append(math.cos(self.cal_elementwise_angle(coord, cur_freq)))
        # embed: shape (input_embed_dim)
        return embed

    def cal_input_dim(self):
        # compute the dimention of the encoded spatial relation embedding
        return int(self.coord_dim * self.frequency_num * 2)

    def cal_freq_list(self):
        # if self.freq_init == "random":
        #     # the frequence we use for each block, alpha in ICLR paper
        #     # self.freq_list shape: (frequency_num)
        #     self.freq_list = np.random.random(size=[self.frequency_num]) * self.max_radius
        # elif self.freq_init == "geometric":
        #     self.freq_list = []
        #     for cur_freq in range(self.frequency_num):
        #         base = 1.0/(np.power(self.max_radius, cur_freq*1.0/(self.frequency_num-1)))
        #         self.freq_list.append(base)

        #     self.freq_list = np.asarray(self.freq_list)
        self.freq_list = _cal_freq_list(self.freq_init, self.frequency_num, self.max_radius, self.min_radius)

    def cal_freq_mat(self):
        # freq_mat shape: (frequency_num, 1)
        freq_mat = np.expand_dims(self.freq_list, axis=1)
        # self.freq_mat shape: (frequency_num, 2)
        self.freq_mat = np.repeat(freq_mat, 2, axis=1)

    def make_input_embeds(self, coords):
        if type(coords) == np.ndarray:
            assert self.coord_dim == np.shape(coords)[2]
            coords = list(coords)
        elif type(coords) == list:
            assert self.coord_dim == len(coords[0][0])
        else:
            raise Exception("Unknown coords data type for GridCellSpatialRelationEncoder")

        # coords_mat: shape (batch_size, num_context_pt, 2)
        coords_mat = np.asarray(coords).astype(float)
        batch_size = coords_mat.shape[0]
        num_context_pt = coords_mat.shape[1]
        # coords_mat: shape (batch_size, num_context_pt, 2, 1)
        coords_mat = np.expand_dims(coords_mat, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, 1, 1)
        coords_mat = np.expand_dims(coords_mat, axis=4)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 1)
        coords_mat = np.repeat(coords_mat, self.frequency_num, axis=3)
        # coords_mat: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        coords_mat = np.repeat(coords_mat, 2, axis=4)
        # spr_embeds: shape (batch_size, num_context_pt, 2, frequency_num, 2)
        spr_embeds = coords_mat * self.freq_mat

        # # convert to radius
        # coords_mat = coords_mat*math.pi/180

        # make sinuniod function
        # sin for 2i, cos for 2i+1
        # spr_embeds: (batch_size, num_context_pt, 2*frequency_num*2=input_embed_dim)
        spr_embeds[:, :, :, :, 0::2] = np.sin(spr_embeds[:, :, :, :, 0::2])  # dim 2i
        spr_embeds[:, :, :, :, 1::2] = np.cos(spr_embeds[:, :, :, :, 1::2])  # dim 2i+1

        # (batch_size, num_context_pt, 2*frequency_num*2)
        spr_embeds = np.reshape(spr_embeds, (batch_size, num_context_pt, -1))

        return spr_embeds

    def forward(self, tensor):
        """
        Given a list of coords (deltaX, deltaY), give their spatial relation embedding
        Args:
            tensor: input features to encode with shape (batch_size, num_context_pt, coord_dim)
        Return:
            sprenc: input tesnor added to the encoding ->Tensor shape (batch_size, num_context_pt, spa_embed_dim)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 3d!")
        batch_size, x, orig_ch = tensor.shape

        pos = torch.arange(float(x ** 0.5), )
        paired = torch.cartesian_prod(pos, pos)

        paired = paired.unsqueeze(0)
        coords = paired.repeat((batch_size, 1, 1))
        assert tuple(coords.shape) == (batch_size, x, 2), 'shape of coordinates is not as expected '
        # print(coords.shape)  # expected [batch_size, num_context_pts,2]

        # Relative coordinates :
        rel_coord = torch.empty((batch_size, x * x, 2))

        center_coord = coords[:, ((x) - 1) // 2, :]  # shape [batch, 1 , 2]
        # print(center_coord,center_coord.shape)
        for i in range(x):  # num of context points
            coord = coords[:, i, :]

            rel_coord[:, i, :] = coord - center_coord
        # print('relative coordinates:',rel_coord[0],rel_coord.shape)
        assert tuple(rel_coord.shape) == (batch_size, x, 2), 'shape of relative coordinates is not as expected'

        # spr_embeds = self.make_input_embeds(coords.numpy())
        spr_embeds = self.make_input_embeds(rel_coord.numpy())

        # # loop over all batches
        # spr_embeds = []
        # for cur_batch in coords:
        #     # loop over N context points
        #     cur_embeds = []
        #     for coords_tuple in cur_batch:
        #         cur_embeds.append(self.cal_coord_embed(coords_tuple))
        #     spr_embeds.append(cur_embeds)

        # spr_embeds: shape (batch_size, num_context_pt, input_embed_dim)
        spr_embeds = torch.FloatTensor(spr_embeds).to(self.device)

        # sprenc: shape (batch_size, num_context_pt, spa_embed_dim)
        # sprenc = torch.einsum("bnd,dk->bnk", (spr_embeds, self.post_mat))
        # sprenc = self.f_act(self.dropout(self.post_linear(spr_embeds)))

        if self.ffn is not None:
            return self.ffn(spr_embeds)
        else:

            # spr_embeds = rearrange(spr_embeds, 'b (p1 p2) d -> b p1 p2 d', p1=x,
            #                       p2=y)
            print('embed shape', spr_embeds.shape)
            return spr_embeds + tensor
