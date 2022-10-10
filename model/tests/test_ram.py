import pytest

from methods.swin_w_ram import ReconfiguredAttentionModule
from test_helper_fns import *

# defaults
B = 4
H = 64
W = 64
C = 96
DET_TOKEN_NUM = 100
INTER_TOKEN_NUM=100
IN_CHANNELS = 96
WINDOW_SIZE = 8
NUM_HEADS = 8



def create_ram(dim, window_size, num_heads, det_token_num, inter_token_num, **kwargs):
    ram = ReconfiguredAttentionModule(dim, window_size, num_heads, **kwargs)
    ram.det_token_num = det_token_num
    ram.inter_token_num=inter_token_num
    return ram


@pytest.fixture
def typical_x():
    return generate_x((B, H, W, C))

@pytest.fixture
def typical_det_inter_tokens():
    return generate_x((B, DET_TOKEN_NUM, C))


def test_RAM(typical_x, typical_det_inter_tokens):
    ram = create_ram(IN_CHANNELS, WINDOW_SIZE, NUM_HEADS, DET_TOKEN_NUM, INTER_TOKEN_NUM)
    
    patch_out, det_out, inter_out = ram(typical_x, typical_det_inter_tokens, typical_det_inter_tokens)
    assert list(patch_out.shape)==list(typical_x.shape)
    assert list(det_out.shape) == list(typical_det_inter_tokens.shape)
    assert list(inter_out.shape) == list(typical_det_inter_tokens.shape)
    
def test_RAM_xs_window(typical_x, typical_det_inter_tokens):
    WINDOW_SIZE = 1
    ram = create_ram(IN_CHANNELS, WINDOW_SIZE, NUM_HEADS, DET_TOKEN_NUM, INTER_TOKEN_NUM)
    
    patch_out, det_out, inter_out = ram(typical_x, typical_det_inter_tokens, typical_det_inter_tokens)
    assert list(patch_out.shape)==list(typical_x.shape)
    assert list(det_out.shape) == list(typical_det_inter_tokens.shape)
    assert list(inter_out.shape) == list(typical_det_inter_tokens.shape)
    
def test_RAM_large_window(typical_x, typical_det_inter_tokens):
    WINDOW_SIZE = 32
    ram = create_ram(IN_CHANNELS, WINDOW_SIZE, NUM_HEADS, DET_TOKEN_NUM, INTER_TOKEN_NUM)
    
    patch_out, det_out, inter_out = ram(typical_x, typical_det_inter_tokens, typical_det_inter_tokens)
    assert list(patch_out.shape)==list(typical_x.shape)
    assert list(det_out.shape) == list(typical_det_inter_tokens.shape)
    assert list(inter_out.shape) == list(typical_det_inter_tokens.shape)
    
def test_RAM_xs_nheads(typical_x, typical_det_inter_tokens):
    NUM_HEADS = 1
    ram = create_ram(IN_CHANNELS, WINDOW_SIZE, NUM_HEADS, DET_TOKEN_NUM, INTER_TOKEN_NUM)
    
    patch_out, det_out, inter_out = ram(typical_x, typical_det_inter_tokens, typical_det_inter_tokens)
    assert list(patch_out.shape)==list(typical_x.shape)
    assert list(det_out.shape) == list(typical_det_inter_tokens.shape)
    assert list(inter_out.shape) == list(typical_det_inter_tokens.shape)
    
def test_RAM_large_nheads(typical_x, typical_det_inter_tokens):
    NUM_HEADS = 48
    ram = create_ram(IN_CHANNELS, WINDOW_SIZE, NUM_HEADS, DET_TOKEN_NUM, INTER_TOKEN_NUM)
    
    patch_out, det_out, inter_out = ram(typical_x, typical_det_inter_tokens, typical_det_inter_tokens)
    assert list(patch_out.shape)==list(typical_x.shape)
    assert list(det_out.shape) == list(typical_det_inter_tokens.shape)
    assert list(inter_out.shape) == list(typical_det_inter_tokens.shape)
    
def test_RAM_patch_det_cross_attn(typical_x, typical_det_inter_tokens):
    ram = create_ram(IN_CHANNELS, WINDOW_SIZE, NUM_HEADS, DET_TOKEN_NUM, INTER_TOKEN_NUM)
    cross_attn_mask = generate_mask_all1((B, NUM_HEADS, DET_TOKEN_NUM, H * W + DET_TOKEN_NUM))
    x = (typical_x, typical_x)
    patch_out, det_out, inter_out = ram(x, typical_det_inter_tokens, typical_det_inter_tokens, 
                                        cross_attn=True, cross_attn_mask=cross_attn_mask)
    assert list(patch_out.shape)==list(typical_x.shape)
    assert list(det_out.shape) == list(typical_det_inter_tokens.shape)
    assert list(inter_out.shape) == list(typical_det_inter_tokens.shape)
    
def test_RAM_inter_det_cross_attn(typical_x, typical_det_inter_tokens):
    ram = create_ram(IN_CHANNELS, WINDOW_SIZE, NUM_HEADS, DET_TOKEN_NUM, INTER_TOKEN_NUM)
    # cross_attn_mask = generate_mask_all1((B, NUM_HEADS, DET_TOKEN_NUM, H * W + DET_TOKEN_NUM))
    
    patch_out, det_out, inter_out = ram(typical_x, typical_det_inter_tokens, typical_det_inter_tokens, 
                                        inter_det_cross_attn=True)
    assert list(patch_out.shape)==list(typical_x.shape)
    assert list(det_out.shape) == list(typical_det_inter_tokens.shape)
    assert list(inter_out.shape) == list(typical_det_inter_tokens.shape)
    
def test_RAM_patch_det_inter_det_cross_attn(typical_x, typical_det_inter_tokens):
    ram = create_ram(IN_CHANNELS, WINDOW_SIZE, NUM_HEADS, DET_TOKEN_NUM, INTER_TOKEN_NUM)
    # cross_attn_mask = generate_mask_all1((B, NUM_HEADS, DET_TOKEN_NUM, H * W + DET_TOKEN_NUM))
    cross_attn_mask = generate_mask_all1((B, NUM_HEADS, DET_TOKEN_NUM, H * W + DET_TOKEN_NUM))
    x = (typical_x, typical_x)
    patch_out, det_out, inter_out = ram(x, typical_det_inter_tokens, typical_det_inter_tokens, 
                                        cross_attn=True, cross_attn_mask=cross_attn_mask,
                                        inter_det_cross_attn=True)
    assert list(patch_out.shape)==list(typical_x.shape)
    assert list(det_out.shape) == list(typical_det_inter_tokens.shape)
    assert list(inter_out.shape) == list(typical_det_inter_tokens.shape)
    
def test_RAM_patch_inter_cross_attn(typical_x, typical_det_inter_tokens):
    INTER_TOKEN_NUM=300
    inter_tokens = generate_x((B, INTER_TOKEN_NUM, C))
    ram = create_ram(IN_CHANNELS, WINDOW_SIZE, NUM_HEADS, DET_TOKEN_NUM, INTER_TOKEN_NUM)
    inter_cross_attn_mask = generate_mask_all1((B, NUM_HEADS, INTER_TOKEN_NUM, H * W + INTER_TOKEN_NUM))
    x = (typical_x, typical_x)
    patch_out, det_out, inter_out = ram(x, typical_det_inter_tokens, inter_tokens, 
                                        inter_patch_cross_attn=True, 
                                        inter_patch_cross_attn_mask=inter_cross_attn_mask)
    assert list(patch_out.shape)==list(typical_x.shape)
    assert list(det_out.shape) == list(typical_det_inter_tokens.shape)
    assert list(inter_out.shape) == list(inter_tokens.shape)
    
def test_RAM_patch_det_patch_inter_cross_attn(typical_x, typical_det_inter_tokens):
    INTER_TOKEN_NUM=300
    inter_tokens = generate_x((B, INTER_TOKEN_NUM, C))
    ram = create_ram(IN_CHANNELS, WINDOW_SIZE, NUM_HEADS, DET_TOKEN_NUM, INTER_TOKEN_NUM)
    cross_attn_mask = generate_mask_all1((B, NUM_HEADS, DET_TOKEN_NUM, H * W + DET_TOKEN_NUM))
    inter_cross_attn_mask = generate_mask_all1((B, NUM_HEADS, INTER_TOKEN_NUM, H * W + INTER_TOKEN_NUM))
    x = (typical_x, typical_x)
    patch_out, det_out, inter_out = ram(x, typical_det_inter_tokens, inter_tokens, 
                                        cross_attn=True, cross_attn_mask=cross_attn_mask,
                                        inter_patch_cross_attn=True, 
                                        inter_patch_cross_attn_mask=inter_cross_attn_mask)
    assert list(patch_out.shape)==list(typical_x.shape)
    assert list(det_out.shape) == list(typical_det_inter_tokens.shape)
    assert list(inter_out.shape) == list(inter_tokens.shape)
    
def test_RAM_inter_det_patch_inter_cross_attn(typical_x, typical_det_inter_tokens):
    INTER_TOKEN_NUM=300
    inter_tokens = generate_x((B, INTER_TOKEN_NUM, C))
    ram = create_ram(IN_CHANNELS, WINDOW_SIZE, NUM_HEADS, DET_TOKEN_NUM, INTER_TOKEN_NUM)
    # cross_attn_mask = generate_mask_all1((B, NUM_HEADS, DET_TOKEN_NUM, H * W + DET_TOKEN_NUM))
    inter_cross_attn_mask = generate_mask_all1((B, NUM_HEADS, INTER_TOKEN_NUM, H * W + INTER_TOKEN_NUM + DET_TOKEN_NUM))
    x = (typical_x, typical_x)
    patch_out, det_out, inter_out = ram(x, typical_det_inter_tokens, inter_tokens, 
                                        inter_det_cross_attn=True,
                                        inter_patch_cross_attn=True, 
                                        inter_patch_cross_attn_mask=inter_cross_attn_mask)
    assert list(patch_out.shape)==list(typical_x.shape)
    assert list(det_out.shape) == list(typical_det_inter_tokens.shape)
    assert list(inter_out.shape) == list(inter_tokens.shape)
    
def test_RAM_patch_det_inter_det_patch_inter_cross_attn(typical_x, typical_det_inter_tokens):
    INTER_TOKEN_NUM=300
    inter_tokens = generate_x((B, INTER_TOKEN_NUM, C))
    ram = create_ram(IN_CHANNELS, WINDOW_SIZE, NUM_HEADS, DET_TOKEN_NUM, INTER_TOKEN_NUM)
    cross_attn_mask = generate_mask_all1((B, NUM_HEADS, DET_TOKEN_NUM, H * W + DET_TOKEN_NUM))
    inter_cross_attn_mask = generate_mask_all1((B, NUM_HEADS, INTER_TOKEN_NUM, H * W + INTER_TOKEN_NUM+DET_TOKEN_NUM))
    x = (typical_x, typical_x)
    patch_out, det_out, inter_out = ram(x, typical_det_inter_tokens, inter_tokens, 
                                        inter_det_cross_attn=True,
                                        cross_attn=True, cross_attn_mask=cross_attn_mask,
                                        inter_patch_cross_attn=True, 
                                        inter_patch_cross_attn_mask=inter_cross_attn_mask)
    assert list(patch_out.shape)==list(typical_x.shape)
    assert list(det_out.shape) == list(typical_det_inter_tokens.shape)
    assert list(inter_out.shape) == list(inter_tokens.shape)