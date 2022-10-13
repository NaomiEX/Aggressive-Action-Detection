import pytest
import torch

from methods.vidt.deformable_transformer import DeformableTransformerDecoderLayer\
    , InteractionLayer, DeformableTransformerDecoder, DeformableTransformer
from test_helper_fns import *

# defaults
DIM_MODEL = 256
DIM_FFN = 1024
DIM_FEATS=256
DROPOUT=0.1
NLEVELS = 4
NHEADS = 8
NPOINTS = 4
B=4
DET_TOKEN_NUM=100
INTER_TOKEN_NUM=100
POS_DIM=256
ORI_H = 640
ORI_W = 640
NUM_DECODER_LAYERS=6

def create_deform_trans_layer(d_model=DIM_MODEL, d_ffn=DIM_FFN,
                                dropout=DROPOUT,
                                n_levels=NLEVELS,
                                n_heads=NHEADS,
                                n_points=NPOINTS):
    return DeformableTransformerDecoderLayer(d_model=d_model,
                                            d_ffn=d_ffn,
                                            dropout=dropout,
                                            n_levels=n_levels,
                                            n_heads=n_heads,
                                            n_points=n_points)


def create_inter_layer(d_model=DIM_MODEL, d_feats=DIM_FEATS, dropout=DROPOUT):
    return InteractionLayer(d_model, d_feats, dropout=dropout)
    
def get_spatial_shapes_per_layer(ori_h=ORI_H, ori_w=ORI_W, nlevels=NLEVELS):
    return [[ori_h/(8<<i), ori_w/(8<<i)] for i in range(nlevels)]

def create_input(det_token_num=DET_TOKEN_NUM, model_dim=DIM_MODEL,
                ori_h=ORI_H, ori_w=ORI_W, nlevels=NLEVELS,ref_points_ver=4):
    tgt = generate_x((B, det_token_num, model_dim)).to('cuda')
    query_pos = generate_x((B, det_token_num, model_dim)).to('cuda')
    if ref_points_ver == 4:  
        ref_points = generate_x(B, det_token_num, 4, 4).to('cuda')
    else :
        ref_points = generate_x(B, det_token_num, 2).to('cuda')
    spatial_dims_per_layer = get_spatial_shapes_per_layer(nlevels=nlevels, ori_h=ori_h, ori_w=ori_w)
    combined = int(sum([h*w for h, w in spatial_dims_per_layer]))
    src= generate_x((B, combined, model_dim)).to('cuda')
    src_spatial_shapes = torch.Tensor(spatial_dims_per_layer).type(torch.long).to('cuda')
    combined_tens = torch.Tensor([h*w for h, w in spatial_dims_per_layer])
    cumulative = combined_tens.cumsum(0)
    for i in range(cumulative.shape[-1]-1, 0, -1):
        cumulative[i]=cumulative[i-1]
    cumulative[0]=0
    level_start_idx = cumulative.type(torch.long).to('cuda')
    return tgt, query_pos, ref_points, src, src_spatial_shapes, level_start_idx


def test_deform_trans_decoder_layer():
    layer = create_deform_trans_layer()
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input()
    print(src.shape)
    print(src_spatial_shapes.shape)
    print(level_start_index.shape)
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    
def test_deform_trans_decoder_layer_large_dmodel():
    DIM_MODEL = 1024
    layer = create_deform_trans_layer(d_model=DIM_MODEL)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input(model_dim=DIM_MODEL)
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    
def test_deform_trans_decoder_layer_small_dmodel():
    DIM_MODEL = 16
    layer = create_deform_trans_layer(d_model=DIM_MODEL)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input(model_dim=DIM_MODEL)
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    print(tgt.shape)
    
def test_deform_trans_decoder_layer_large_dffn():
    DIM_FFN = 8192
    layer = create_deform_trans_layer(d_ffn=DIM_FFN)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input()
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    print(out.shape)
    
def test_deform_trans_decoder_layer_small_dffn():
    DIM_FFN = 16
    layer = create_deform_trans_layer(d_ffn=DIM_FFN)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input()
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    
def test_deform_trans_decoder_layer_dropout_all():
    layer = create_deform_trans_layer(dropout=1.0)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input()
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    
def test_deform_trans_decoder_layer_large_nheads():
    
    layer = create_deform_trans_layer(n_heads=64)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input()
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    print(out.shape)
    
def test_deform_trans_decoder_layer_min_nheads():
    
    layer = create_deform_trans_layer(n_heads=1)
    layer.to('cuda')
    tgt, query_pos, ref_points, src\
            , src_spatial_shapes, level_start_index = create_input()
    out = layer(tgt, query_pos,ref_points,
            src, src_spatial_shapes, level_start_index)
    assert list(out.shape) == list(tgt.shape)
    

def test_inter_layer():
    DET_TOKEN_NUM=100
    INTER_TOKEN_NUM=100
    det = generate_x((B, DET_TOKEN_NUM, DIM_MODEL))
    inter = generate_x((B, INTER_TOKEN_NUM, DIM_MODEL))
    inter_layer = create_inter_layer()
    out = inter_layer(det, inter)
    assert len(out) == 2
    det_out, inter_out = out
    assert list(det_out.shape) == list(det.shape)
    assert list(inter_out.shape) == list(inter.shape)
    
def test_inter_layer_inequal_det_inter():
    DET_TOKEN_NUM=100
    INTER_TOKEN_NUM=300
    det = generate_x((B, DET_TOKEN_NUM, DIM_MODEL))
    inter = generate_x((B, INTER_TOKEN_NUM, DIM_MODEL))
    inter_layer = create_inter_layer()
    out = inter_layer(det, inter)
    assert len(out) == 2
    det_out, inter_out = out
    assert list(det_out.shape) == list(det.shape)
    assert list(inter_out.shape) == list(inter.shape)
    
def test_inter_layer_small_dmodel():
    DIM_MODEL=1
    DET_TOKEN_NUM=100
    INTER_TOKEN_NUM=100
    det = generate_x((B, DET_TOKEN_NUM, DIM_MODEL))
    inter = generate_x((B, INTER_TOKEN_NUM, DIM_MODEL))
    inter_layer = create_inter_layer(d_model=DIM_MODEL, d_feats=DIM_MODEL)
    out = inter_layer(det, inter)
    assert len(out) == 2
    det_out, inter_out = out
    assert list(det_out.shape) == list(det.shape)
    assert list(inter_out.shape) == list(inter.shape)
    
def test_inter_layer_large_dmodel():
    DIM_MODEL=512
    DET_TOKEN_NUM=100
    INTER_TOKEN_NUM=100
    det = generate_x((B, DET_TOKEN_NUM, DIM_MODEL))
    inter = generate_x((B, INTER_TOKEN_NUM, DIM_MODEL))
    inter_layer = create_inter_layer(d_model=DIM_MODEL, d_feats=DIM_MODEL)
    out = inter_layer(det, inter)
    assert len(out) == 2
    det_out, inter_out = out
    assert list(det_out.shape) == list(det.shape)
    assert list(inter_out.shape) == list(inter.shape)

def create_decoder_input(det_token_num=DET_TOKEN_NUM, inter_token_num=INTER_TOKEN_NUM, 
                            model_dim=DIM_MODEL,
                ori_h=ORI_H, ori_w=ORI_W, nlevels=NLEVELS):
    tgt, query_pos, ref_points, src\
        , src_spatial_shapes, level_start_idx = create_input(det_token_num=det_token_num,
                                                                model_dim=model_dim,
                                                                ori_h=ori_h, ori_w=ori_w, nlevels=nlevels,
                                                                ref_points_ver=2)
    src_valid_ratios = generate_x((B, NLEVELS, 2)).cuda()
    inter_tgt = generate_x((B, inter_token_num, model_dim)).cuda()
    inter_query_pos = generate_x((B, inter_token_num, model_dim)).to('cuda')
    sub_ref_points = ref_points
    src_padding_mask = generate_mask_all0((B, src.shape[1])).bool().cuda()
    return tgt, inter_tgt, query_pos, inter_query_pos, ref_points, sub_ref_points\
        , src, src_spatial_shapes, level_start_idx, src_valid_ratios, src_padding_mask


def test_deform_trans_decoder():
    decoder_layer = create_deform_trans_layer()
    inter_decoder_layer = create_deform_trans_layer()
    inter_layer = create_inter_layer()
    decoder = DeformableTransformerDecoder(decoder_layer, inter_decoder_layer, inter_layer,
                                            num_layers=NUM_DECODER_LAYERS).cuda()
    tgt, inter_tgt, query_pos, inter_query_pos, ref_points, sub_ref_points\
        , src, src_spatial_shapes, level_start_idx\
            , src_valid_ratios, src_padding_mask = create_decoder_input()
    
    out = decoder(tgt, inter_tgt, ref_points, sub_ref_points, src, src_spatial_shapes, 
                    level_start_idx, src_valid_ratios,
                    query_pos=query_pos,inter_query_pos=inter_query_pos,
                    src_padding_mask=src_padding_mask)
    assert len(out) == 4
    

def test_deform_trans_decoder():
    decoder_layer = create_deform_trans_layer()
    inter_decoder_layer = create_deform_trans_layer()
    inter_layer = create_inter_layer()
    decoder = DeformableTransformerDecoder(decoder_layer, inter_decoder_layer, inter_layer,
                                            num_layers=NUM_DECODER_LAYERS).cuda()
    tgt, inter_tgt, query_pos, inter_query_pos, ref_points, sub_ref_points\
        , src, src_spatial_shapes, level_start_idx\
            , src_valid_ratios, src_padding_mask = create_decoder_input()
    
    out = decoder(tgt, inter_tgt, ref_points, sub_ref_points, src, src_spatial_shapes, 
                    level_start_idx, src_valid_ratios,
                    query_pos=query_pos,inter_query_pos=inter_query_pos,
                    src_padding_mask=src_padding_mask)
    assert len(out) == 4
    det_out, inter_out, ref_points_out, sub_ref_points_out = out
    assert list(det_out.shape) == list(tgt.shape)
    assert list(inter_out.shape) == list(inter_tgt.shape)
    assert list(ref_points_out.shape) == list(ref_points.shape)
    assert list(sub_ref_points_out.shape) == list(sub_ref_points.shape)
    

def test_deform_trans_decoder_interm():
    decoder_layer = create_deform_trans_layer()
    inter_decoder_layer = create_deform_trans_layer()
    inter_layer = create_inter_layer()
    decoder = DeformableTransformerDecoder(decoder_layer, inter_decoder_layer, inter_layer,
                                            num_layers=NUM_DECODER_LAYERS, return_intermediate=True).cuda()
    tgt, inter_tgt, query_pos, inter_query_pos, ref_points, sub_ref_points\
        , src, src_spatial_shapes, level_start_idx\
            , src_valid_ratios, src_padding_mask = create_decoder_input()
    
    out = decoder(tgt, inter_tgt, ref_points, sub_ref_points, src, src_spatial_shapes, 
                    level_start_idx, src_valid_ratios,
                    query_pos=query_pos,inter_query_pos=inter_query_pos,
                    src_padding_mask=src_padding_mask)
    assert len(out) == 4
    det_out, inter_out, ref_points_out, sub_ref_points_out = out
    assert list(det_out.shape) == [NUM_DECODER_LAYERS+1] + list(tgt.shape)
    assert list(inter_out.shape) == [NUM_DECODER_LAYERS+1] + list(inter_tgt.shape)
    assert list(ref_points_out.shape) == [NUM_DECODER_LAYERS+1] + list(ref_points.shape)
    assert list(sub_ref_points_out.shape) == [NUM_DECODER_LAYERS+1] + list(sub_ref_points.shape)
    
    