import pytest
from torch import nn
from math import ceil

from methods.swin_w_ram import PatchEmbed
from test_model import generate_x


def create_patch_embed(patch_size, embed_dim, **kwargs):
    return PatchEmbed(patch_size=patch_size, embed_dim=embed_dim, norm_layer=nn.LayerNorm, **kwargs)


def test_patch_split_dims_4x4_small_equal_divisible():
    B=4
    W=64
    H=64
    C=3
    PATCH_SIZE=4
    EMBED_DIM=48
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, H/4, W/4]
    
def test_patch_split_dims_4x4_large_equal_divisible():
    B=4
    W=4000
    H=4000
    C=3
    PATCH_SIZE=4
    EMBED_DIM=48
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, H/4, W/4]
    
def test_patch_split_dims_4x4_small_equal_indivisible():
    B=4
    W=37
    H=37
    C=3
    PATCH_SIZE=4
    EMBED_DIM=48
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, ceil(H/4), ceil(W/4)]
    
def test_patch_split_dims_4x4_xsmall_equal_indivisible():
    B=4
    W=2
    H=2
    C=3
    PATCH_SIZE=4
    EMBED_DIM=48
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, ceil(H/4), ceil(W/4)]
    
def test_patch_split_dims_4x4_patchsize_equal_indivisible():
    B=4
    W=4
    H=4
    C=3
    PATCH_SIZE=4
    EMBED_DIM=48
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, 1, 1]
    
def test_patch_split_dims_4x4_zero():
    B=4
    W=0
    H=0
    C=3
    PATCH_SIZE=4
    EMBED_DIM=48
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    with pytest.raises(ValueError):
        patch_embed(x)
        
def test_patch_split_dims_4x4_small_inequal_divisible():
    B=4
    W=40
    H=20
    C=3
    PATCH_SIZE=4
    EMBED_DIM=48
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, H/4, W/4]
        
def test_patch_split_dims_4x4_small_inequal_indivisible():
    B=4
    W=37
    H=43
    C=3
    PATCH_SIZE=4
    EMBED_DIM=48
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, ceil(H/4), ceil(W/4)]
    
def test_patch_split_dims_8x8():
    B=4
    W=64
    H=64
    C=3
    PATCH_SIZE=8
    EMBED_DIM=48
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, H/PATCH_SIZE, W/PATCH_SIZE]
    
def test_patch_split_dims_1x1():
    B=4
    W=64
    H=64
    C=3
    PATCH_SIZE=1
    EMBED_DIM=48
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, H/PATCH_SIZE, W/PATCH_SIZE]
    
def test_patch_split_0x0():
    PATCH_SIZE=0
    EMBED_DIM=48
    with pytest.raises(ValueError):
        create_patch_embed(PATCH_SIZE, EMBED_DIM)
        
def test_patch_split_dims_embed_large():
    B=4
    W=64
    H=64
    C=3
    PATCH_SIZE=4
    EMBED_DIM=96
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, H/PATCH_SIZE, W/PATCH_SIZE]
    
def test_patch_split_dims_embed_xs():
    B=4
    W=64
    H=64
    C=3
    PATCH_SIZE=4
    EMBED_DIM=1
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, H/PATCH_SIZE, W/PATCH_SIZE]

def test_patch_split_embed_0():
    PATCH_SIZE=1
    EMBED_DIM=0
    with pytest.raises(ValueError):
        create_patch_embed(PATCH_SIZE, EMBED_DIM)


def test_patch_split_ch_1():
    B=4
    W=64
    H=64
    C=1
    PATCH_SIZE=4
    EMBED_DIM=48
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM, in_chans=C)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, H/PATCH_SIZE, W/PATCH_SIZE]
    
def test_patch_split_weight_standardization():
    B=4
    W=64
    H=64
    C=3
    PATCH_SIZE=4
    EMBED_DIM=48
    
    x = generate_x((B, C, H, W))
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM, weight_standardization=True)
    assert list(patch_embed(x).shape) == [B, EMBED_DIM, H/PATCH_SIZE, W/PATCH_SIZE]