from torch import nn
import pytest
from methods.swin_w_ram import SwinTransformer, swin_nano
from test_helper_fns import *

# defaults
PRETRAIN_IMG_SIZE = [224, 224]
EMBED_DIM=48
DEPTHS=[2,2,6,2]
NUM_HEADS=[3,6,12,24]
WINDOW_SIZE=7
DROP_PATH_RATE=0.0
METHOD='vidt'
DET_TOKEN_NUM=100
INTER_TOKEN_NUM=300
POS_DIM=256
CROSS_INDICES=[]
INTER_CROSS_INDICES=[]
INTER_PATCH_CROSS_INDICES=[]
B=4
C=3
H=64
W=64

def get_swin_transformer(pretrain_img_size=PRETRAIN_IMG_SIZE,
                        embed_dim=EMBED_DIM,
                        depths=DEPTHS,
                        num_heads=NUM_HEADS,
                        window_size=WINDOW_SIZE,
                        drop_path_rate=DROP_PATH_RATE,
                        det_token_num=DET_TOKEN_NUM,
                        inter_token_num=INTER_TOKEN_NUM, 
                        cross_indices=CROSS_INDICES,
                        inter_cross_indices=INTER_CROSS_INDICES,
                        inter_patch_cross_indices=INTER_PATCH_CROSS_INDICES):
    swin=SwinTransformer(pretrain_img_size=pretrain_img_size,
                    embed_dim=embed_dim,
                    depths=depths,
                    num_heads=num_heads,
                    window_size=window_size,
                    drop_path_rate=drop_path_rate)
    # swin=swin_nano()[0]
    swin.finetune_det(METHOD, det_token_num=det_token_num,
                        inter_token_num=inter_token_num, 
                        cross_indices=cross_indices, 
                        inter_cross_indices=inter_cross_indices,
                        inter_patch_cross_indices=inter_patch_cross_indices)
    return swin

@pytest.fixture
def typical_x():
    return generate_x((B, C, H, W))

@pytest.fixture
def trivial_mask():
    return generate_mask_all0((B, H, W))

@pytest.fixture
def typical_swin_transformer_output(typical_x, trivial_mask):
    swin = get_swin_transformer()
    out = swin(typical_x, trivial_mask)
    return out

def test_swin_transformer_patch(typical_swin_transformer_output):
    num_layers=len(DEPTHS)
    assert len(typical_swin_transformer_output) == 5
    patch_outs, _, _, _, _ = typical_swin_transformer_output
    assert len(patch_outs)==num_layers
    for i in range(num_layers-1):
        assert list(patch_outs[i].shape) == [B, EMBED_DIM<<(i+1), H/(8<<i), W/(8<<i)]
    assert list(patch_outs[-1].shape) == [B, POS_DIM, H/(8<<(num_layers-1)), W/(8<<(num_layers-1)) ]
    
def test_swin_transformer_det(typical_swin_transformer_output):
    num_layers=len(DEPTHS)
    assert len(typical_swin_transformer_output) == 5
    _, det_outs, _, _, _ = typical_swin_transformer_output
    assert list(det_outs.shape) == [B, EMBED_DIM << (num_layers-1), DET_TOKEN_NUM]
    
def test_swin_transformer_inter(typical_swin_transformer_output):
    num_layers=len(DEPTHS)
    assert len(typical_swin_transformer_output) == 5
    _, _, inter_outs, _, _ = typical_swin_transformer_output
    assert list(inter_outs.shape) == [B, EMBED_DIM << (num_layers-1), INTER_TOKEN_NUM]
    
def test_swin_transformer_det_pos(typical_swin_transformer_output):
    assert len(typical_swin_transformer_output) == 5
    _, _, _, det_pos, _ = typical_swin_transformer_output
    assert list(det_pos.shape)==[1, POS_DIM, DET_TOKEN_NUM]
    
def test_swin_transformer_inter_pos(typical_swin_transformer_output):
    assert len(typical_swin_transformer_output) == 5
    _, _, _, _, inter_pos = typical_swin_transformer_output
    assert list(inter_pos.shape)==[1, POS_DIM, INTER_TOKEN_NUM]
    
    
def test_swin_transformer_xattn(typical_x, trivial_mask):
    CROSS_INDICES=[3]
    swin = get_swin_transformer(cross_indices=CROSS_INDICES)
    out = swin(typical_x, trivial_mask)
    num_layers=len(DEPTHS)
    assert len(out) == 5
    patch_outs, _, _, _, _ = out
    assert len(patch_outs)==num_layers
    for i in range(num_layers-1):
        assert list(patch_outs[i].shape) == [B, EMBED_DIM<<(i+1), H/(8<<i), W/(8<<i)]
    assert list(patch_outs[-1].shape) == [B, POS_DIM, H/(8<<(num_layers-1)), W/(8<<(num_layers-1)) ]
    
def test_swin_transformer_xattn_all(typical_x, trivial_mask):
    CROSS_INDICES=[0, 1, 2, 3]
    swin = get_swin_transformer(cross_indices=CROSS_INDICES)
    out = swin(typical_x, trivial_mask)
    num_layers=len(DEPTHS)
    assert len(out) == 5
    patch_outs, _, _, _, _ = out
    assert len(patch_outs)==num_layers
    for i in range(num_layers-1):
        assert list(patch_outs[i].shape) == [B, EMBED_DIM<<(i+1), H/(8<<i), W/(8<<i)]
    assert list(patch_outs[-1].shape) == [B, POS_DIM, H/(8<<(num_layers-1)), W/(8<<(num_layers-1)) ]
    
    
def test_swin_transformer_det_inter_xattn(typical_x, trivial_mask):
    INTER_CROSS_INDICES=[3]
    swin = get_swin_transformer(inter_cross_indices=INTER_CROSS_INDICES)
    out = swin(typical_x, trivial_mask)
    num_layers=len(DEPTHS)
    assert len(out) == 5
    patch_outs, _, _, _, _ = out
    assert len(patch_outs)==num_layers
    for i in range(num_layers-1):
        assert list(patch_outs[i].shape) == [B, EMBED_DIM<<(i+1), H/(8<<i), W/(8<<i)]
    assert list(patch_outs[-1].shape) == [B, POS_DIM, H/(8<<(num_layers-1)), W/(8<<(num_layers-1)) ]
    
def test_swin_transformer_det_inter_xattn_all(typical_x, trivial_mask):
    INTER_CROSS_INDICES=[0, 1, 2, 3]
    swin = get_swin_transformer(inter_cross_indices=INTER_CROSS_INDICES)
    out = swin(typical_x, trivial_mask)
    num_layers=len(DEPTHS)
    assert len(out) == 5
    patch_outs, _, _, _, _ = out
    assert len(patch_outs)==num_layers
    for i in range(num_layers-1):
        assert list(patch_outs[i].shape) == [B, EMBED_DIM<<(i+1), H/(8<<i), W/(8<<i)]
    assert list(patch_outs[-1].shape) == [B, POS_DIM, H/(8<<(num_layers-1)), W/(8<<(num_layers-1)) ]
    
def test_swin_transformer_patch_inter_xattn(typical_x, trivial_mask):
    INTER_PATCH_CROSS_INDICES=[3]
    swin = get_swin_transformer(inter_patch_cross_indices=INTER_PATCH_CROSS_INDICES)
    out = swin(typical_x, trivial_mask)
    num_layers=len(DEPTHS)
    assert len(out) == 5
    patch_outs, _, _, _, _ = out
    assert len(patch_outs)==num_layers
    for i in range(num_layers-1):
        assert list(patch_outs[i].shape) == [B, EMBED_DIM<<(i+1), H/(8<<i), W/(8<<i)]
    assert list(patch_outs[-1].shape) == [B, POS_DIM, H/(8<<(num_layers-1)), W/(8<<(num_layers-1)) ]
    
def test_swin_transformer_patch_inter_xattn_all(typical_x, trivial_mask):
    INTER_PATCH_CROSS_INDICES=[0, 1, 2, 3]
    swin = get_swin_transformer(inter_patch_cross_indices=INTER_PATCH_CROSS_INDICES)
    out = swin(typical_x, trivial_mask)
    num_layers=len(DEPTHS)
    assert len(out) == 5
    patch_outs, _, _, _, _ = out
    assert len(patch_outs)==num_layers
    for i in range(num_layers-1):
        assert list(patch_outs[i].shape) == [B, EMBED_DIM<<(i+1), H/(8<<i), W/(8<<i)]
    assert list(patch_outs[-1].shape) == [B, POS_DIM, H/(8<<(num_layers-1)), W/(8<<(num_layers-1)) ]
    
def test_swin_transformer_patch_det_inter_det_xattn(typical_x, trivial_mask):
    CROSS_INDICES=[3]
    INTER_CROSS_INDICES=[3]
    swin = get_swin_transformer(cross_indices=CROSS_INDICES, inter_cross_indices=INTER_CROSS_INDICES)
    out = swin(typical_x, trivial_mask)
    num_layers=len(DEPTHS)
    assert len(out) == 5
    patch_outs, _, _, _, _ = out
    assert len(patch_outs)==num_layers
    for i in range(num_layers-1):
        assert list(patch_outs[i].shape) == [B, EMBED_DIM<<(i+1), H/(8<<i), W/(8<<i)]
    assert list(patch_outs[-1].shape) == [B, POS_DIM, H/(8<<(num_layers-1)), W/(8<<(num_layers-1)) ]
    
def test_swin_transformer_patch_det_patch_inter_xattn(typical_x, trivial_mask):
    CROSS_INDICES=[3]
    INTER_PATCH_CROSS_INDICES=[3]
    swin = get_swin_transformer(cross_indices=CROSS_INDICES, inter_patch_cross_indices=INTER_PATCH_CROSS_INDICES)
    out = swin(typical_x, trivial_mask)
    num_layers=len(DEPTHS)
    assert len(out) == 5
    patch_outs, _, _, _, _ = out
    assert len(patch_outs)==num_layers
    for i in range(num_layers-1):
        assert list(patch_outs[i].shape) == [B, EMBED_DIM<<(i+1), H/(8<<i), W/(8<<i)]
    assert list(patch_outs[-1].shape) == [B, POS_DIM, H/(8<<(num_layers-1)), W/(8<<(num_layers-1)) ]

def test_swin_transformer_inter_det_patch_inter_xattn(typical_x, trivial_mask):
    INTER_CROSS_INDICES=[3]
    INTER_PATCH_CROSS_INDICES=[3]
    swin = get_swin_transformer(inter_cross_indices=INTER_CROSS_INDICES, 
                                inter_patch_cross_indices=INTER_PATCH_CROSS_INDICES)
    out = swin(typical_x, trivial_mask)
    num_layers=len(DEPTHS)
    assert len(out) == 5
    patch_outs, _, _, _, _ = out
    assert len(patch_outs)==num_layers
    for i in range(num_layers-1):
        assert list(patch_outs[i].shape) == [B, EMBED_DIM<<(i+1), H/(8<<i), W/(8<<i)]
    assert list(patch_outs[-1].shape) == [B, POS_DIM, H/(8<<(num_layers-1)), W/(8<<(num_layers-1)) ]
    
def test_swin_transformer_patch_det_inter_det_patch_inter_xattn(typical_x, trivial_mask):
    CROSS_INDICES=[3]
    INTER_CROSS_INDICES=[3]
    INTER_PATCH_CROSS_INDICES=[3]
    swin = get_swin_transformer(cross_indices=CROSS_INDICES,
                                inter_cross_indices=INTER_CROSS_INDICES, 
                                inter_patch_cross_indices=INTER_PATCH_CROSS_INDICES)
    out = swin(typical_x, trivial_mask)
    num_layers=len(DEPTHS)
    assert len(out) == 5
    patch_outs, _, _, _, _ = out
    assert len(patch_outs)==num_layers
    for i in range(num_layers-1):
        assert list(patch_outs[i].shape) == [B, EMBED_DIM<<(i+1), H/(8<<i), W/(8<<i)]
    assert list(patch_outs[-1].shape) == [B, POS_DIM, H/(8<<(num_layers-1)), W/(8<<(num_layers-1)) ]
    
def test_swin_transformer_patch_det_inter_det_patch_inter_xattn_all(typical_x, trivial_mask):
    CROSS_INDICES=[0, 1, 2, 3]
    INTER_CROSS_INDICES=[0, 1, 2, 3]
    INTER_PATCH_CROSS_INDICES=[0, 1, 2, 3]
    swin = get_swin_transformer(cross_indices=CROSS_INDICES,
                                inter_cross_indices=INTER_CROSS_INDICES, 
                                inter_patch_cross_indices=INTER_PATCH_CROSS_INDICES)
    out = swin(typical_x, trivial_mask)
    num_layers=len(DEPTHS)
    assert len(out) == 5
    patch_outs, _, _, _, _ = out
    assert len(patch_outs)==num_layers
    for i in range(num_layers-1):
        assert list(patch_outs[i].shape) == [B, EMBED_DIM<<(i+1), H/(8<<i), W/(8<<i)]
    assert list(patch_outs[-1].shape) == [B, POS_DIM, H/(8<<(num_layers-1)), W/(8<<(num_layers-1)) ]
    
