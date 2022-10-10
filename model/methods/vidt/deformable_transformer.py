# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------

import copy
import math
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_, normal_

from util.misc import inverse_sigmoid
from ops.modules import MSDeformAttn

from timm.models.layers import DropPath


class DeformableTransformer(nn.Module):
    """ A Deformable Transformer for the neck in a detector

    The transformer encoder is completely removed for ViDT
    Parameters:
        d_model: the channel dimension for attention [default=256]
        nhead: the number of heads [default=8]
        num_decoder_layers: the number of decoding layers [default=6]
        dim_feedforward: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        return_intermediate_dec: whether to return all the indermediate outputs [default=True]
        num_feature_levels: the number of scales for extracted features [default=4]
        dec_n_points: the number of reference points for deformable attention [default=4]
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
        token_label: whether to use the token label loss for training [default=False]. This is an additional trick
            proposed in  https://openreview.net/forum?id=LhbD74dsZFL (ICLR'22) for further improvement
    """

    def __init__(self, d_model=256, nhead=8, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu", return_intermediate_dec=True, num_feature_levels=4, dec_n_points=4,
                 drop_path=0., token_label=False, instance_aware_attn=True,
                 single_branch=False, det_token_num=100, inter_token_num=100):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.det_token_num = det_token_num
        self.inter_token_num = inter_token_num
        decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points,
                                                        drop_path=drop_path)
        inter_decoder_layer = DeformableTransformerDecoderLayer(d_model, dim_feedforward,
                                                        dropout, activation,
                                                        num_feature_levels, nhead, dec_n_points,
                                                        drop_path=drop_path)
        instance_aware_attn = InteractionLayer(d_model, d_model, dropout) if instance_aware_attn else None
            
        self.decoder = DeformableTransformerDecoder(decoder_layer, inter_decoder_layer, 
                                                    instance_aware_attn, num_decoder_layers, 
                                                    return_intermediate_dec, single_branch=single_branch)
        

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        self.token_label = token_label

        self.reference_points = nn.Linear(d_model, 2)
        self.sub_reference_points = nn.Linear(d_model, 2)

        if self.token_label:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)

            self.token_embed = nn.Linear(d_model, 91)
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.token_embed.bias.data = torch.ones(91) * bias_value

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, tgt, inter_tgt, query_pos, inter_query_pos):
        """ The forward step of the decoder

        Parameters:
            srcs: [Patch] tokens from the last 4 layers of the Swin backbone, 
                    list of length 4 with shapes [B, 256, H/8, W/8],
                    [B, 256, H/16, W/16], [B, 256, H/32, W/32], [B, 256, H/64, W/64]
            masks: input padding mask from the last 4 layers of the Swin backbone,
                    list of length 4 with shapes [B, H/8, W/8], [B, H/16, W/16], ...
            tgt: [DET] tokens, shape: [B, 100, 256]
            inter_tgt: [INTER] tokens, shape: [B, 100, 256]
            query_pos: [DET] token pos encodings

        Returns:
            hs: calibrated [DET] tokens
            init_reference_out: init reference points
            inter_references_out: intermediate reference points for box refinement
            enc_token_class_unflat: info. for token labeling
        """

        # prepare input for the Transformer decoder
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []
        for lvl, (src, mask) in enumerate(zip(srcs, masks)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        memory = src_flatten
        bs, _, c = memory.shape
        tgt = tgt # [DET] tokens, shape: [B, 100, 256]
        query_pos = query_pos.expand(bs, -1, -1) # [DET] token pos encodings
        inter_query_pos = inter_query_pos.expand(bs, -1, -1)

        # prepare input for token label
        if self.token_label:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
        enc_token_class_unflat = None
        if self.token_label:
            enc_token_class = self.token_embed(output_memory)
            enc_token_class_unflat = []
            for st, (h, w) in zip(level_start_index, spatial_shapes):
                enc_token_class_unflat.append(enc_token_class[:, st:st+h*w, :].view(bs, h, w, 91))

        # reference points for deformable attention
        reference_points = self.reference_points(query_pos).sigmoid()
        sub_reference_points = self.sub_reference_points(query_pos).sigmoid()
        init_reference_out = reference_points # query_pos -> reference point
        init_sub_reference_out = sub_reference_points
        
        # sub_reference_points = self.sub_reference_points(query_pos).sigmoid()
        # init_sub_reference_out = sub_reference_points

        # decoder
        hs, rels, inter_references, inter_sub_references = self.decoder(tgt, inter_tgt, reference_points, 
                                                                        sub_reference_points, memory, 
                                                                        spatial_shapes, level_start_index, 
                                                                        valid_ratios, query_pos, 
                                                                        inter_query_pos, mask_flatten)

        inter_references_out = inter_references
        inter_sub_references_out = inter_sub_references

        return hs, rels, init_reference_out, init_sub_reference_out, inter_references_out, inter_sub_references_out, enc_token_class_unflat


class DeformableTransformerDecoderLayer(nn.Module):
    """ A decoder layer.

    Parameters:
        d_model: the channel dimension for attention [default=256]
        d_ffn: the channel dim of point-wise FFNs [default=1024]
        dropout: the degree of dropout used in FFNs [default=0.1]
        activation: An activation function to use [default='relu']
        n_levels: the number of scales for extracted features [default=4]
        n_heads: the number of heads [default=8]
        n_points: the number of reference points for deformable attention [default=4]
        drop_path: the ratio of stochastic depth for decoding layers [default=0.0]
    """

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, drop_path=0.):
        super().__init__()

        # [DET x PATCH] deformable cross-attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # [DET x DET] self-attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # # [INTER] x [PATCH] deformable cross-attention
        # self.cross_attn_inter = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        # self.dropout1_inter = nn.Dropout(dropout)
        # self.norm1_inter = nn.LayerNorm(d_model)
        
        # # [INTER] x [INTER] self-attention
        # self.inter_self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        # self.dropout2_inter = nn.Dropout(dropout)
        # self.norm2_inter = nn.LayerNorm(d_model)
        

        # ffn for multi-heaed
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        # stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, 
                src_padding_mask=None, sub_reference_points=None):

        # [DET] self-attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # Multi-scale deformable cross-attention in Eq. (1) in the ViDT paper
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                                reference_points,
                                src, src_spatial_shapes, level_start_index, src_padding_mask)
        print(tgt2)
        if sub_reference_points is not None:
            tgt2_sub = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                                sub_reference_points,
                                src, src_spatial_shapes, level_start_index, src_padding_mask)
            tgt2 = tgt2 + tgt2_sub

        if self.drop_path is None:
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            # ffn
            tgt = self.forward_ffn(tgt)
        else:
            tgt = tgt + self.drop_path(self.dropout1(tgt2))
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
            tgt = tgt + self.drop_path(self.dropout4(tgt2))
            tgt = self.norm3(tgt)

        return tgt
    

class DeformableTransformerDecoder(nn.Module):
    """ A Decoder consisting of multiple layers

    Parameters:
        decoder_layer: a deformable decoding layer
        num_layers: the number of layers
        return_intermediate: whether to return intermediate resutls
    """

    def __init__(self, decoder_layer, inter_decoder_layer, 
                    inter_layer, num_layers, return_intermediate=False,
                    single_branch=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        if single_branch:
            self.inter_decoder_layers = None
            self.inter_layers = None
        else:
            self.inter_decoder_layers = _get_clones(inter_decoder_layer, num_layers)
            self.inter_layers = _get_clones(inter_layer, num_layers) if inter_layer is not None else None
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement
        self.bbox_embed = None
        self.sub_bbox_embed = None
        self.class_embed = None
        
        # whether decoder is single or double branched
        self.single_branch=single_branch
        
    def refine_bbox(self, bbox_inter_pred, reference_points):
        if reference_points.shape[-1] == 4:
            new_reference_points = bbox_inter_pred + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
        else:
            assert reference_points.shape[-1] == 2
            new_reference_points = bbox_inter_pred
            new_reference_points[..., :2] = bbox_inter_pred[..., :2] + inverse_sigmoid(reference_points)
            new_reference_points = new_reference_points.sigmoid()
        reference_points = new_reference_points.detach()
        return reference_points
    
    def refine_sub_obj_bbox(self, index, output, reference_points):
        # NOTE: not used, ignore
        tmp = self.bbox_embed[index](output)
        tmp_sub = self.sub_bbox_embed[index](output)
        
        obj_reference_points = self.refine_bbox(tmp, reference_points)
        sub_reference_points = self.refine_bbox(tmp_sub, reference_points)
        
        reference_points = (obj_reference_points + sub_reference_points) / 2
        return reference_points

    def forward(self, tgt, inter_tgt, reference_points, sub_reference_points, src, 
                src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, inter_query_pos=None, src_padding_mask=None):
        """ The forwared step of the Deformable Decoder

        Parameters:
            tgt: [DET] tokens
            reference_poitns: reference points for deformable attention
            src: the [PATCH] tokens fattened into a 1-d sequence
            src_spatial_shapes: the spatial shape of each multi-scale feature map
            src_level_start_index: the start index to refer different scale inputs
            src_valid_ratios: the ratio of multi-scale feature maps
            query_pos: the pos encoding for [DET] tokens
            src_padding_mask: the input padding mask

        Returns:
            output: [DET] tokens calibrated (i.e., object embeddings)
            reference_points: A reference points

            If return_intermediate = True, output & reference_points are returned from all decoding layers
        """

        output = tgt # shape: [B, 100, 256]
        inter_output = inter_tgt # shape: [B, 100, 256]
        intermediate = []
        intermediate_rel = []
        intermediate_reference_points = []
        intermediate_sub_reference_points = []

        if self.bbox_embed is not None:
            tmp = self.bbox_embed[0](output)
            reference_points = self.refine_bbox(tmp, reference_points)
        if self.sub_bbox_embed is not None:
            tmp_sub = self.sub_bbox_embed[0](output)
            sub_reference_points = self.refine_bbox(tmp_sub, sub_reference_points)
            

        if self.return_intermediate:
            intermediate.append(output)
            intermediate_rel.append(inter_output)
            intermediate_reference_points.append(reference_points)
            intermediate_sub_reference_points.append(sub_reference_points)
            
        for i in range(self.num_layers):
            instance_decoder_layer = self.layers[i]
            if not self.single_branch:
                interaction_decoder_layer = self.inter_decoder_layers[i]
            if self.inter_layers is not None:
                instance_aware_attn_layer = self.inter_layers[i]
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                sub_reference_points_input = sub_reference_points[:, :, None] \
                                        * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]  
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
                
                sub_reference_points_input = sub_reference_points[:, :, None] * src_valid_ratios[:, None]
                
            if self.single_branch:
                # start = time.time()
                output, inter_output = instance_decoder_layer(output, query_pos,inter_output, inter_query_pos, 
                                                reference_points_input, src, src_spatial_shapes, 
                                                src_level_start_index, src_padding_mask, 
                                                sub_reference_points_input)
            else:
                # output shape; [B, 100, 256]
                # start = time.time()
                output = instance_decoder_layer(output, query_pos, reference_points_input, src, src_spatial_shapes, 
                                                src_level_start_index, src_padding_mask, sub_reference_points_input)
            
            # output shape; [B, 100, 256]
            if not self.single_branch:
                # start2 = time.time()
                inter_output = interaction_decoder_layer(inter_output, inter_query_pos, reference_points_input, 
                                                        src, src_spatial_shapes, src_level_start_index, src_padding_mask)
            
            if self.inter_layers is not None:
                output, inter_output = instance_aware_attn_layer(output, inter_output)
            
            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                # reference_points = self.refine_sub_obj_bbox(i+1, output, reference_points)
                tmp = self.bbox_embed[i+1](output)
                reference_points = self.refine_bbox(tmp, reference_points)
                
            if self.sub_bbox_embed is not None:
                tmp_sub = self.sub_bbox_embed[i+1](output)
                sub_reference_points = self.refine_bbox(tmp_sub, sub_reference_points)
                
            #

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_rel.append(inter_output)
                intermediate_reference_points.append(reference_points)
                intermediate_sub_reference_points.append(sub_reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_rel), torch.stack(intermediate_reference_points), torch.stack(intermediate_sub_reference_points)

        return output, inter_output, reference_points, sub_reference_points


class InteractionLayer(nn.Module):
    """
    Instance-aware attention layer
    """
    def __init__(self, d_model, d_feature, dropout=0.1):
        # defaults
        #   d_model = 256;  d_feature = 256;  dropout = 0.1

        super().__init__()
        self.d_feature = d_feature

        self.det_tfm = nn.Linear(d_model, d_feature)
        self.rel_tfm = nn.Linear(d_model, d_feature)
        self.det_value_tfm = nn.Linear(d_model, d_feature)

        self.rel_norm = nn.LayerNorm(d_model)

        # if use dropout
        if dropout is not None:
            self.dropout = dropout
            self.det_dropout = nn.Dropout(dropout)
            self.rel_add_dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, det_in, rel_in):
        """
        det_in: output of instance branch's i-th decoder layer (shape: [B, 100, 256])
        rel_in: output of interaction branch's i-th decoder layre (shape: [B, 100, 256])
        """ 
        det_attn_in = self.det_tfm(det_in)
        rel_attn_in = self.rel_tfm(rel_in)
        det_value = self.det_value_tfm(det_in) # [B, 100, 256]

        # affinity score map (analogous to Q-K mapping)
        # result shape: [B, 100 (det), 100 (inter)]
        scores = torch.matmul(det_attn_in,
            rel_attn_in.transpose(1, 2)) / math.sqrt(self.d_feature)

        # softmax across the last dimension,
        det_weight = F.softmax(scores.transpose(1, 2), dim = -1) # shape: [B, 100 (inter), 100 (det)]

        if self.dropout is not None:
          det_weight = self.det_dropout(det_weight)
        
        rel_add = torch.matmul(det_weight, det_value) # shape: [B, 100 (inter), 256]
        rel_out = self.rel_add_dropout(rel_add) + rel_in
        rel_out = self.rel_norm(rel_out)

        return det_in, rel_out

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""

    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_deforamble_transformer(args):
    use_iaa = not args.without_iaa
    return DeformableTransformer(
        d_model=args.reduced_dim,
        nhead=args.nheads,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        token_label=args.token_label,
        instance_aware_attn=use_iaa,
        single_branch=args.single_branch_decoder,
        det_token_num=args.det_token_num, 
        inter_token_num=args.inter_token_num)


