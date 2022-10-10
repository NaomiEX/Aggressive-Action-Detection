# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# --------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from collections.abc import Iterable

from util.misc import Conv2d


class Mlp(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if (self.fc1.in_features != x.shape[-1]):
            raise ValueError("Expected input features={self.fc1.in_features}, got {x.shape[-1]}")
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def masked_sin_pos_encoding(x, mask, num_pos_feats, temperature=10000, scale=2 * math.pi):
    """ Masked Sinusoidal Positional Encoding

    Parameters:
        x: [PATCH] tokens
        mask: the padding mask for [PATCH] tokens
        num_pos_feats: the size of channel dimension
        temperature: the temperature value
        scale: the normalization scale

    Returns:
        pos: Sinusoidal positional encodings
    """

    num_pos_feats = num_pos_feats // 2
    not_mask = ~mask

    y_embed = not_mask.cumsum(1, dtype=torch.float32)
    x_embed = not_mask.cumsum(2, dtype=torch.float32)

    eps = 1e-6
    y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3)

    return pos


def window_partition(x, window_size):
    """
    Parameters:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    if (H//window_size <1 or W//window_size <1):
        raise ValueError(f"H and W must be >= window_size but window_size={window_size} and H={H}, W={W}")
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Parameters:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    if (windows.shape[1] != window_size or windows.shape[2]!=window_size):
        raise ValueError(f"second and third dimension of input must match window size," \
            "instead got input.shape={windows.shape} and window_shape={window_size}")
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class ReconfiguredAttentionModule(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias -> extended with RAM.
    It supports both of shifted and non-shifted window.

    !!!!!!!!!!! IMPORTANT !!!!!!!!!!!
    The original attention module in Swin is replaced with the reconfigured attention module in Section 3.
    All the parameters are shared, so only the forward function is modified.
    See https://arxiv.org/pdf/2110.03921.pdf
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Parameters:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, 
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        if (not isinstance(window_size, Iterable)):
            window_size = to_2tuple(window_size)
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, det, inter, mask=None, cross_attn=False, 
                inter_det_cross_attn = False, 
                cross_attn_mask=None, inter_patch_cross_attn=False, 
                inter_patch_cross_attn_mask=None):
        """ Forward function.
        RAM module receives [Patch] and [DET] tokens and returns their calibrated ones

        Parameters:
            x: [PATCH] tokens, [B, H, W, Layer_C]
            det: [DET] tokens, [B, 100, Layer_C]
            inter: [INTER] tokens, [B, 100, Layer_C]
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None -> mask for shifted window attention

            "additional inputs for RAM"
            cross_attn: whether to use cross-attention [det x patch] (for selective cross-attention)
            inter_det_cross_attn: whether to use [inter x det] cross-attention
            cross_attn_mask: mask for cross-attention
            inter_patch_cross_attn: whether to use [inter x patch] cross-attention
            inter_patch_cross_attn_mask: mask for [inter x [patch + inter]] or [inter x [patch + det + inter]]

        Returns:
            patch_x: the calibrated [PATCH] tokens
            det_x: the calibrated [DET] tokens
        """
        assert self.window_size[0] == self.window_size[1]
        window_size = self.window_size[0]
        local_map_size = window_size * window_size

        # projection before window partitioning
        if not cross_attn and not inter_patch_cross_attn:
            B, H, W, C = x.shape
            N = H * W
            x = x.view(B, N, C)
            x = torch.cat([x, det, inter], dim=1) # t.s. [B, H*W + 100 + 100, 48]
            full_qkv = self.qkv(x) # t.s. [B, H*W + 100 + 100, 144]
            patch_qkv, det_qkv, inter_qkv = full_qkv[:, :N, :], full_qkv[:, N:N+self.det_token_num, :], full_qkv[:, -self.inter_token_num:, :]
        else:
            B, H, W, C = x[0].shape # x is a tuple(shifted_x, cross_path); shape of shifted window version
            N = H * W
            _, ori_H, ori_W, _ = x[1].shape # x is a tuple(shifted_x, cross_path); shape of original x
            ori_N = ori_H * ori_W

            shifted_x = x[0].view(B, N, C)
            cross_x = x[1].view(B, ori_N, C)
            x = torch.cat([shifted_x, cross_x, det, inter], dim=1)
            full_qkv = self.qkv(x)
            patch_qkv, cross_patch_qkv, det_qkv, inter_qkv = \
                full_qkv[:, :N, :], full_qkv[:, N:N + ori_N, :], full_qkv[:, N + ori_N:N+ori_N+self.det_token_num, :], full_qkv[:,-self.inter_token_num:, :]
        patch_qkv = patch_qkv.view(B, H, W, -1) # shape: [B, H, W, 144]

        # window partitioning for [PATCH] tokens
        patch_qkv = window_partition(patch_qkv, window_size) # shape: [B * num_windows, 7, 7, 144]
        B_ = patch_qkv.shape[0]
        patch_qkv = patch_qkv.reshape(B_, window_size * window_size, 3, self.num_heads, C // self.num_heads)
        _patch_qkv = patch_qkv.permute(2, 0, 3, 1, 4)
        patch_q, patch_k, patch_v = _patch_qkv[0], _patch_qkv[1], _patch_qkv[2] # each one has shape: [B*num_windows, num_heads, 7*7, 16]

        # [PATCH x PATCH] self-attention using window partitions
        patch_q = patch_q * self.scale
        patch_attn = (patch_q @ patch_k.transpose(-2, -1)) # shape: [B*num_windows, num_heads, 7*7, 7*7]
        # add relative pos bias for [patch x patch] self-attention
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # shape: [7*7, 7*7, num_heads]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        # same relative position bias for every window for every image
        patch_attn = patch_attn + relative_position_bias.unsqueeze(0) # shape: [B*num_windows, num_heads, 7*7, 7*7]

        # if shifted window is used, it needs to apply the mask
        if mask is not None:
            nW = mask.shape[0] # [num_heads, 49, 49]
            # patch_attn = [B, num_windows, num_heads, 49, 49]
            patch_attn = patch_attn.view(B_ // nW, nW, self.num_heads, local_map_size, local_map_size) + \
                         mask.unsqueeze(1).unsqueeze(0)
            patch_attn = patch_attn.view(-1, self.num_heads, local_map_size, local_map_size) # [B*num_windows, num_heads, 49, 49]

        patch_attn = self.softmax(patch_attn)
        patch_attn = self.attn_drop(patch_attn)
        patch_x = (patch_attn @ patch_v).transpose(1, 2).reshape(B_, window_size, window_size, C)

        # extract qkv for [DET] tokens
        det_qkv = det_qkv.view(B, -1, 3, self.num_heads, C // self.num_heads) # shape: [B, 100, 3, nH, 48/nH]
        det_qkv = det_qkv.permute(2, 0, 3, 1, 4)
        det_q, det_k, det_v = det_qkv[0], det_qkv[1], det_qkv[2] # each has shape: [B, 3, 100, 16]
        det_k_orig = det_k
        det_v_orig = det_v
        # if cross-attention is activated
        
        if cross_attn or inter_patch_cross_attn:

            # reconstruct the spatial form of [PATCH] tokens for global [DET x PATCH] attention
            cross_patch_qkv = cross_patch_qkv.view(B, ori_H, ori_W, 3, self.num_heads, C // self.num_heads)
            patch_kv = cross_patch_qkv[:, :, :, 1:, :, :].permute(3, 0, 4, 1, 2, 5).contiguous()
            patch_kv = patch_kv.view(2, B, self.num_heads, ori_H * ori_W, -1)

            # extract "key and value" of [PATCH] tokens for cross-attention
            cross_patch_k, cross_patch_v = patch_kv[0], patch_kv[1]

        if cross_attn:
            # bind key and value of [PATCH] and [DET] tokens for [DET X [PATCH, DET]] attention
            det_k, det_v = torch.cat([cross_patch_k, det_k], dim=2), torch.cat([cross_patch_v, det_v], dim=2)

        # [DET x DET] self-attention or binded [DET x [PATCH, DET]] attention
        det_q = det_q * self.scale
        det_attn = (det_q @ det_k.transpose(-2, -1)) # [B, 3, 100, 100]
        # apply cross-attention mask if available
        if cross_attn_mask is not None:
            det_attn = det_attn + cross_attn_mask
        det_attn = self.softmax(det_attn)
        det_attn = self.attn_drop(det_attn)
        det_x = (det_attn @ det_v).transpose(1, 2).reshape(B, -1, C) # [B, 100, 48]
        
        # extract query, key, value for [INTER] tokens
        inter_qkv = inter_qkv.view(B, -1, 3, self.num_heads, C // self.num_heads) # shape: [B, 100, 3, num_heads, 16]
        inter_qkv = inter_qkv.permute(2, 0, 3, 1, 4) # shape: [3, B, num_heads, 100, 16]
        inter_q, inter_k, inter_v = inter_qkv[0], inter_qkv[1], inter_qkv[2] # [B, 3, 100, 16]
        
        if inter_det_cross_attn:
        # if cross_attn:
            # shape: [B, 8, 100+100, C]
            inter_k, inter_v = torch.cat([det_k_orig, inter_k], dim=2), torch.cat([det_v_orig, inter_v], dim=2)
        if inter_patch_cross_attn:
            # print("inter patch")
            inter_k, inter_v = torch.cat([cross_patch_k, inter_k], dim=2), torch.cat([cross_patch_v, inter_v], dim=2)
            
        # [INTER x INTER] self-attention 
        inter_q = inter_q * self.scale
        # [B, 8, 100, C] matmul [B,8,C, 100+100] = [B, 8,100, 200]
        inter_attn = (inter_q @ inter_k.transpose(-2, -1)) # shape: [B, num_heads, 100, 100]
        # apply cross-attention mask if available
        if inter_patch_cross_attn_mask is not None:
            inter_attn = inter_attn + inter_patch_cross_attn_mask
        inter_attn = self.softmax(inter_attn)
        inter_attn = self.attn_drop(inter_attn)
        # [B, 3, 100, 200] matmul [B, 3, 200, C] = [B, 3, 100, C]
        inter_x = (inter_attn @ inter_v).transpose(1, 2).reshape(B, -1, C) # [B, 100, 48]

        # reverse window for [PATCH] tokens <- the output of [PATCH x PATCH] self attention
        patch_x = window_reverse(patch_x, window_size, H, W) # shape: [B, H, W, 48]

        # projection for outputs from multi-head
        x = torch.cat([patch_x.view(B, H*W, C), det_x, inter_x], dim=1) # shape: [B, H*W + 100 + 100, 48]
        x = self.proj(x)
        x = self.proj_drop(x)

        # decompose after FFN into [PATCH] and [DET] tokens
        patch_x = x[:, :H * W, :].view(B, H, W, C)
        det_x = x[:, H * W: H * W + self.det_token_num, :]
        inter_x = x[:, -self.inter_token_num:, :]

        return patch_x, det_x, inter_x


class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block.

    Parameters:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = ReconfiguredAttentionModule(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.H = None
        self.W = None

    def forward(self, x, mask_matrix, pos, cross_attn, inter_cross_attn, cross_attn_mask, 
                inter_patch_cross_attn=False, inter_patch_cross_attn_mask=None):
        """ Forward function.

        Parameters:
            x: Input feature, tensor size (B, H*W + DET + INTER, C). i.e., binded [PATCH, DET, INTER] tokens
            H, W: Spatial resolution of the input feature.
            mask_matrix: Attention mask for cyclic shift, shape: [num_windows, 49, 49]

            "additional inputs'
            pos: (patch_pos, det_pos, inter_pos)
                patch_pos: sine pos encoding of [PATCH] tokens if cross attention, otherwise None
                det_pos: learnable [DET] token positional embeddings,[1, 100, 256]
                inter_pos: learnable [INTER] token positional embeddings, [1, 100, 256]
            cross_attn: whether to use cross attn [det x [det + patch]]
            inter_cross_attn: whether to use [inter x det] cross attn
            cross_attn_mask: attention mask for cross-attention
            inter_patch_cross_attn: whether to use [inter x patch] cross attn
            inter_patch_cross_attn_mask: mask for [inter x [patch+inter]] or [inter x [patch+det+inter]] cross attn

        Returns:
            x: calibrated & binded [PATCH, DET] tokens
        """

        B, L, C = x.shape
        H, W = self.H, self.W

        assert L == H * W + self.det_token_num + self.inter_token_num, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        # x.shape: [B, H*W, 48]
        # det.shape: [B, 100, 48]
        # inter.shape: [B, 100, 48]
        x, det, inter = x[:, :H * W, :], x[:, H * W:(H*W)+self.det_token_num, :], x[:, -self.inter_token_num:, :]
        x = x.view(B, H, W, C) # shape: [B, H, W, 48]
        orig_x = x

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # projection for det positional encodings: make the channel size suitable for the current layer
        patch_pos, det_pos, inter_pos = pos
        # det_inter_pos = self.det_inter_pos_linear(det_inter_pos)
        # linearly project from positional embedding dimension (256) to layer dimension (48, 96, ...)
        det_pos = self.det_pos_linear(det_pos) # [1, 100, 48]
        inter_pos = self.inter_pos_linear(inter_pos) # [1, 100, 48]

        # cyclic shift
        if self.shift_size > 0:
            # shift the tensor by 3 values to the left, 3 to the top
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # prepare cross-attn and add positional encodings
        if cross_attn or inter_patch_cross_attn:
            # patch token (for cross-attention) + Sinusoidal pos encoding
            cross_patch = orig_x + patch_pos
            # det token + learnable pos encoding
            det = det + det_pos
            inter = inter + inter_pos
            shifted_x = (shifted_x, cross_patch)
        else:
            # it cross_attn is deativated, only [PATCH] and [DET] self-attention are performed
            det = det + det_pos
            inter = inter + inter_pos
            shifted_x = shifted_x

        # W-MSA/SW-MSA
        shifted_x, det, inter = self.attn(shifted_x, mask=attn_mask,
                                   # additional parameters
                                   det=det, inter=inter,
                                   cross_attn=cross_attn, inter_det_cross_attn=inter_cross_attn,
                                   cross_attn_mask=cross_attn_mask, inter_patch_cross_attn=inter_patch_cross_attn,
                                   inter_patch_cross_attn_mask=inter_patch_cross_attn_mask)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)
        x = torch.cat([x, det, inter], dim=1)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Parameters:
        dim (int): Number of input channels.
    """

    def __init__(self, dim, expand=True):
        super().__init__()
        
        self.dim = dim
        norm_layer = nn.LayerNorm
        # if expand is True, the channel size will be expanded, otherwise, return 256 size of channel
        expand_dim = 2 * dim if expand else 256
        self.reduction = nn.Linear(4 * dim, expand_dim, bias=False)
        self.norm = norm_layer(4 * dim)

        # added for detection token [please ignore, not used for training]
        # not implemented yet.
        self.expansion = nn.Linear(dim, expand_dim, bias=False)
        self.norm2 = norm_layer(dim)

    def forward(self, x, H, W):
        """ Forward function.

        Parameters:
            x: Input feature, tensor size (B, H*W, C), i.e., binded [PATCH, DET] tokens
            H, W: Spatial resolution of the input feature.

        Returns:
            x: merged [PATCH, DET] tokens;
            only [PATCH] tokens are reduced in spatial dim, while [DET] tokens is fix-scale
        """

        B, L, C = x.shape
        assert L == H * W + self.det_token_num +self.inter_token_num, "input feature has wrong size"

        x, det, inter = x[:, :H * W, :], x[:, H * W:H*W + self.det_token_num, :], x[:, -self.inter_token_num:, :]
        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # simply repeating for DET and INTER tokens
        det = det.repeat(1, 1, 4)
        
        inter = inter.repeat(1, 1, 4)

        x = torch.cat([x, det, inter], dim=1)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Parameters:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 last=False,
                 use_checkpoint=False):

        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.depth = depth
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, expand=(not last))
        else:
            self.downsample = None


    def forward(self, x, H, W, det_pos, inter_pos, input_mask, 
                cross_attn=False, inter_cross_attn=False,
                inter_patch_cross_attn=False):
        """ Forward function.

        Parameters:
            x: Input feature consists of [PATCH], [DET], [INTER] tokens, tensor size (B, H*W + 100 + 100, C).
            H, W: Spatial resolution of the input feature.
            det_pos: pos encoding for det token, shape: [1, 100, 256]
            inter_pos: pos encoding for inter token, shape: [1, 100, 256]
            input_mask: padding mask for cross attention,
                None if the current layer has no cross attention
            cross_attn: whether to use cross attn [det x [det + patch]]
            inter_cross_attn: whether to use [inter x det] cross attention
            inter_patch_cross_attn: whether to use [inter x patch] cross attention
        """

        B = x.shape[0]

        # calculate attention mask for SW-MSA
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # 1 Hp Wp 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # mask for cyclic shift
        mask_windows = window_partition(img_mask, self.window_size) # shape: [num_windows, 7, 7, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size) # [num_windows, 49]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-math.inf)).masked_fill(attn_mask == 0, float(0.0))

        # compute sinusoidal pos encoding and cross-attn mask here to avoid redundant computation
        if cross_attn or inter_patch_cross_attn:
            _H, _W = input_mask.shape[1:]
            if not (_H == H and _W == W):
                input_mask = F.interpolate(input_mask[None].float(), size=(H, W)).to(torch.bool)[0]

            # sinusoidal pos encoding for [PATCH] tokens used in cross-attention
            patch_pos = masked_sin_pos_encoding(x, input_mask, self.dim)

            # attention padding mask due to the zero padding in inputs
            # the zero (padded) area is masked by 1.0 in 'input_mask'
            cross_attn_mask = input_mask.float()
            cross_attn_mask = cross_attn_mask.masked_fill(cross_attn_mask != 0.0, float(-math.inf)). \
                masked_fill(cross_attn_mask == 0.0, float(0.0)) # [B, 1, 1, H*W]

            # pad for detection token (this padding is required to process the binded [PATCH, DET] attention
            cross_attn_mask = cross_attn_mask.view(B, H * W).unsqueeze(1).unsqueeze(2)
            if cross_attn:
                det_patch_cross_attn_mask = F.pad(cross_attn_mask, (0, self.det_token_num), value=0)
                
            
            if inter_patch_cross_attn:
                if inter_cross_attn: # [INTER] x [INTER, DET, PATCH]
                    inter_patch_cross_attn_mask = F.pad(cross_attn_mask, (0, self.det_token_num + self.inter_token_num), value=0)
                else: # [INTER] x [INTER, PATCH]
                    inter_patch_cross_attn_mask = F.pad(cross_attn_mask, (0, self.inter_token_num), value=0)
        else: # no cross attention
            patch_pos = None
            
        if not cross_attn:
            det_patch_cross_attn_mask = None
        if not inter_patch_cross_attn:
            inter_patch_cross_attn_mask = None
            

        # zip pos encodings
        pos = (patch_pos, det_pos, inter_pos)

        for n_blk, blk in enumerate(self.blocks):
            blk.H, blk.W = H, W

            # for selective cross-attention
            # if cross_attn:
            #     _cross_attn = True
            #     _cross_attn_mask = cross_attn_mask
            #     _pos = pos # i.e., (patch_pos, det_pos, inter_pos)
            # else:
            #     _cross_attn = False
            #     _cross_attn_mask = None
            #     _pos = (None, det_pos, inter_pos)
                

            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask,
                                          # additional inputs
                                          pos=pos,
                                          cross_attn=cross_attn, inter_cross_attn = inter_cross_attn,
                                          cross_attn_mask=det_patch_cross_attn_mask, inter_patch_cross_attn=inter_patch_cross_attn,
                                          inter_patch_cross_attn_mask=inter_patch_cross_attn_mask)
            else:
                x = blk(x, attn_mask,
                        # additional inputs
                        pos=pos,
                        cross_attn=cross_attn, inter_cross_attn=inter_cross_attn,
                        cross_attn_mask=det_patch_cross_attn_mask, 
                        inter_patch_cross_attn=inter_patch_cross_attn,
                        inter_patch_cross_attn_mask=inter_patch_cross_attn_mask) # shape: [B, H*W, 48]
                # torch.save(x, 'stored/x')
        # reduce the number of patch tokens, but maintaining a fixed-scale det tokens
        # meanwhile, the channel dim increases by a factor of 2
        if self.downsample is not None:
            x_down = self.downsample(x, H, W) # shape: [B, H/2*W/2, 96]
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Parameters:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, weight_standardization=False):
        super().__init__()
        if (patch_size <= 0):
            raise ValueError("patch size must be > 0 but got: {patch_size}")
        if (embed_dim <= 0):
            raise ValueError("embed_dim must be > 0 but got: {embed_dim}")
        
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        

        # aggregates the image in patches of size 4x4, stride=4 ensures that the patches are non-overlapping
        if weight_standardization:
            self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            raise ValueError("self.norm is None!")
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if (H <= 0 or W <= 0):
            raise ValueError(f"Width and Height of x must be > 0 but got width: {W} and height: {H}!")
        
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # shape: [B, 48, H/4, W/4]
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2) # [B, H/4 * W/4, 48]
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww) # [B, 48, H/4, W/4]

        return x


class SwinTransformer(nn.Module):
    """ Swin Transformer backbone.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Parameters:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=[1, 2, 3], # NOTE: not used in the current version, please ignore.
                 frozen_stages=-1,
                 use_checkpoint=False,
                 weight_standardization=False
                 ):
        super().__init__()
        
        print(f"\n WEIGHT STANDARDIZATION ARG IS: {weight_standardization}\n")

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,
            weight_standardization=weight_standardization)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                # modified by ViDT
                downsample=PatchMerging if (i_layer < self.num_layers) else None,
                last=None if (i_layer < self.num_layers-1) else True,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        # Not used in the current version -> please ignore. this error will be fixed later
        # we leave this lines to load the pre-trained model ...
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.ape:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'det_pos_embed', 'inter_pos_embed', 'det_token'}

    def finetune_det(self, method, det_token_num=100, inter_token_num=100, pos_dim=256, 
                     cross_indices=[3],inter_cross_indices=[3], inter_patch_cross_indices=[]):
        """ A funtion to add neccessary (leanable) variables to Swin Transformer for object detection

            Parameters:
                method: vidt or vidt_wo_neck
                det_token_num: the number of object to detect, i.e., number of object queries
                pos_dim: the channel dimension of positional encodings for [DET] and [PATCH] tokens
                cross_indices: the indices where to use the [DET X PATCH] cross-attention
                    there are four possible stages in [0, 1, 2, 3]. 3 indicates Stage 4 in the ViDT paper.
        """

        # which method?
        self.method = method

        # how many object we detect?
        self.det_token_num = det_token_num
        self.det_token = nn.Parameter(torch.zeros(1, det_token_num, self.num_features[0]))
        self.det_token = trunc_normal_(self.det_token, std=.02)
        
        # how many interactions to detect
        self.inter_token_num = inter_token_num
        self.inter_token = nn.Parameter(torch.zeros(1, inter_token_num, self.num_features[0]))
        self.inter_token = trunc_normal_(self.inter_token, std=.02)

        # dim size of pos encoding
        self.pos_dim = pos_dim

        # learnable positional encoding for detection tokens
        # shared with interaction tokens
        # NOTE: (setting 1.1.2)
        # det_inter_pos_embed = torch.zeros(1, det_token_num, pos_dim)
        det_pos_embed = torch.zeros(1, det_token_num, pos_dim)
        inter_pos_embed = torch.zeros(1, inter_token_num, pos_dim)
        
        # det_inter_pos_embed = trunc_normal_(det_inter_pos_embed, std=.02)
        det_pos_embed = trunc_normal_(det_pos_embed, std=.02)
        inter_pos_embed = trunc_normal_(inter_pos_embed, std=.02)
        
        # self.det_inter_pos_embed = torch.nn.Parameter(det_inter_pos_embed)
        self.det_pos_embed = torch.nn.Parameter(det_pos_embed)
        self.inter_pos_embed = torch.nn.Parameter(inter_pos_embed)

        # info for detection
        self.num_channels = [self.num_features[i+1] for i in range(len(self.num_features)-1)]
        if method == 'vidt':
            self.num_channels.append(self.pos_dim) # default: 256 (same to the default pos_dim)
        
        # cross attention indices
        self.cross_indices = cross_indices
        self.inter_cross_indices = inter_cross_indices
        self.inter_patch_cross_indices = inter_patch_cross_indices
        
        # divisor to reduce the spatial size of the mask
        # self.mask_divisor = 2 ** (len(self.layers) - len(self.cross_indices))
        patch_cross_indices = [i for i in cross_indices]
        for idx in inter_patch_cross_indices:
            if idx not in patch_cross_indices:
                patch_cross_indices.append(idx)
                
                
        self.mask_divisors = [(i, 2**i) for i in patch_cross_indices]

        # projection matrix for det pos encoding in each Swin layer (there are 4 blocks)
        for layer in self.layers:
            layer.det_token_num = det_token_num
            layer.inter_token_num = inter_token_num
            if layer.downsample is not None:
                layer.downsample.det_token_num = det_token_num
                layer.downsample.inter_token_num = inter_token_num
            for block in layer.blocks:
                block.det_token_num = det_token_num
                block.inter_token_num = inter_token_num
                # block.det_inter_pos_linear = nn.Linear(pos_dim, block.dim)
                block.det_pos_linear = nn.Linear(pos_dim, block.dim)
                block.inter_pos_linear = nn.Linear(pos_dim, block.dim)
                block.attn.det_token_num = det_token_num
                block.attn.inter_token_num = inter_token_num

        # neck-free model do not require downsamling at the last stage.
        if method == 'vidt_wo_neck':
            self.layers[-1].downsample = None

    def forward(self, x, mask):
        """ Forward function.

            Parameters:
                x: input rgb images, [B, C, H, W]
                mask: input padding masks [0: rgb values, 1: padded values]

            Returns:
                patch_outs: multi-scale [PATCH] tokens (four scales are used)
                    these tokens are the first input of the neck decoder
                det_tgt: final [DET] tokens obtained at the last stage
                    this tokens are the second input of the neck decoder
                det_inter_pos: the learnable pos encoding for [DET] tokens.
                    these encodings are used to generate reference points in deformable attention
        """

        # original input shape
        B, ori_H, ori_W = x.shape[0], x.shape[2], x.shape[3]

        # patch embedding
        x = self.patch_embed(x) # shape: [B, 48, H/4, W/4]

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2) # shape: [B, H/4 * W/4, 48]
        x = self.pos_drop(x)

        ## expand det_token and inter_token for all examples in the batch
        det_token = self.det_token.expand(B, -1, -1) # shape: [B, 100, 48]
        inter_token = self.inter_token.expand(B, -1, -1) # shape: [B, 100, 48]

        # det pos encoding -> will be projected in each block
        # det_inter_pos = self.det_inter_pos_embed
        det_pos = self.det_pos_embed
        inter_pos = self.inter_pos_embed

        # prepare a mask for cross attention
        # mask = F.interpolate(mask[None].float(),
        #              size=(Wh // self.mask_divisor, Ww // self.mask_divisor)).to(torch.bool)[0] # shape: [B, H/32, W/32]
        # masks = [F.interpolate(mask[None].float(), size=(Wh // mask_divisor, Ww//mask_divisor)).to(torch.bool)[0]
        #          for _, mask_divisor in self.mask_divisors]
        masks = {stage: F.interpolate(mask[None].float(), 
                 size=(Wh // mask_divisor, Ww//mask_divisor)).to(torch.bool)[0] 
                 for stage, mask_divisor in self.mask_divisors}
        # mask_stages = [stage for stage, _ in self.mask_divisors]

        patch_outs = []
        for stage in range(self.num_layers):
            layer = self.layers[stage]

            # whether to use cross-attention
            cross_attn = True if stage in self.cross_indices else False
            
            inter_cross_attn = True if stage in self.inter_cross_indices else False
            
            inter_patch_cross_attn = True if stage in self.inter_patch_cross_indices else False

            # concat input
            x = torch.cat([x, det_token, inter_token], dim=1) # shape: [B, H/4*W/4 + 100 + 100, 48]

            if stage in masks.keys():
                stage_mask = masks[stage]
            else:
                stage_mask = None
            
            # inference
            # x is the downsampled one, shape: [B,H/8*W/8 + 100 + 100, 96]
            x_out, H, W, x, Wh, Ww = layer(x, Wh, Ww,
                                           # additional input for VIDT
                                        #    input_mask=mask,
                                            input_mask=stage_mask,
                                           det_pos=det_pos,
                                           inter_pos=inter_pos,
                                           cross_attn=cross_attn, inter_cross_attn=inter_cross_attn,
                                           inter_patch_cross_attn=inter_patch_cross_attn)
            x, det_token, inter_token = x[:, :-(self.det_token_num + self.inter_token_num), :], \
                                        x[:, -(self.det_token_num + self.inter_token_num):-self.inter_token_num, :], \
                                        x[:, -self.inter_token_num:, :]

            # Aggregate intermediate outputs
            if stage > 0:
                patch_out = x_out[:, :-(self.det_token_num + self.inter_token_num), :].view(B, H, W, -1).permute(0, 3, 1, 2)
                patch_outs.append(patch_out)

        # patch token reduced from last stage output
        patch_outs.append(x.view(B, Wh, Ww, -1).permute(0, 3, 1, 2))

        # final det token
        det_tgt = x_out[:, -(self.det_token_num + self.inter_token_num):-self.inter_token_num, :].permute(0, 2, 1)
        
        # final inter token
        inter_tgt = x_out[:, -self.inter_token_num:, :].permute(0, 2, 1) # shape: [1, 256, 100]

        # det token pos encoding
        # det_inter_pos = det_inter_pos.permute(0, 2, 1)
        det_pos = det_pos.permute(0, 2, 1)
        inter_pos = inter_pos.permute(0, 2, 1)

        return patch_outs, det_tgt, inter_tgt, det_pos, inter_pos

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformer, self).train(mode)
        self._freeze_stages()

    # not working in the current version
    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def swin_nano(pretrained=None, **kwargs):
    model = SwinTransformer(pretrain_img_size=[224, 224], embed_dim=48, depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24], window_size=7, drop_path_rate=0.0, **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 384

    if pretrained is not None:
        if pretrained == 'imagenet':
            torch.hub._download_url_to_file(
                    url="https://github.com/naver-ai/vidt/releases/download/v0.1-swin/swin_nano_patch4_window7_224.pth",
                dst="checkpoint.pth"
            )
            checkpoint = torch.load("checkpoint.pth", map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 1K')

        else:
            # print("Model's state_dict:\n\n")
            # for p in model.state_dict():
            #     print(p)
            #     print("\n")
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')

    return model, 384


def swin_tiny(pretrained=None, **kwargs):
    model = SwinTransformer(pretrain_img_size=[224, 224], embed_dim=96, depths=[2, 2, 6, 2],
                            num_heads=[3, 6, 12, 24], window_size=7, drop_path_rate=0.2, **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 768

    if pretrained is not None:
        if pretrained == 'imagenet':
            torch.hub._download_url_to_file(
                url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth",
                dst="checkpoint.pth"
            )
            checkpoint = torch.load("checkpoint.pth", map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 1K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')
    return model, 768


def swin_small(pretrained=None, **kwargs):
    model = SwinTransformer(pretrain_img_size=[224, 224], embed_dim=96, depths=[2, 2, 18, 2],
                            num_heads=[3, 6, 12, 24], window_size=7, drop_path_rate=0.3, **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 768

    if pretrained is not None:
        if pretrained == 'imagenet':
            torch.hub._download_url_to_file(
                url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth",
                dst="checkpoint.pth"
            )
            checkpoint = torch.load("checkpoint.pth", map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 1K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 1K')
    return model, 768


def swin_base_win7(pretrained=None, **kwargs):
    model = SwinTransformer(pretrain_img_size=[224, 224], embed_dim=128, depths=[2, 2, 18, 2],
                            num_heads=[4, 8, 16, 32], window_size=7, drop_path_rate=0.3, **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 1024

    if pretrained is not None:
        if pretrained == 'imagenet':
            torch.hub._download_url_to_file(
                url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth",
                dst="checkpoint.pth"
            )
            checkpoint = torch.load("checkpoint.pth", map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 22K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')
    return model, 1024


def swin_large_win7(pretrained=None, **kwargs):
    model = SwinTransformer(pretrain_img_size=[224, 224], embed_dim=192, depths=[2, 2, 18, 2],
                            num_heads=[6, 12, 24, 48], window_size=7, drop_path_rate=0.3, **kwargs)

    if pretrained is None or pretrained == 'none':
        return model, 1024

    if pretrained is not None:
        if pretrained == 'imagenet':
            torch.hub._download_url_to_file(
                url="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth",
                dst="checkpoint.pth"
            )
            checkpoint = torch.load("checkpoint.pth", map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone pretrained on ImageNet 22K')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            print('Load the backbone in the given path')
    return model, 1024
