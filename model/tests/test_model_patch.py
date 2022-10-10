import pytest
from torch import nn
from math import ceil

from methods.swin_w_ram import PatchEmbed, PatchMerging
from test_helper_fns import generate_x


def create_patch_embed(patch_size, embed_dim, **kwargs):
    return PatchEmbed(patch_size=patch_size, embed_dim=embed_dim, norm_layer=nn.LayerNorm, **kwargs)

def create_patch_merge(dim, det_token_num, inter_token_num, **kwargs):
    pm = PatchMerging(dim, **kwargs)
    pm.det_token_num = det_token_num
    pm.inter_token_num=inter_token_num
    return pm

@pytest.fixture
def typical_x_input():
    B=4
    W=64
    H=64
    C=3
    return generate_x((B, C, H, W))

@pytest.fixture
def typical_x_transformer():
    B=4
    W=64
    H=64
    C=96
    return generate_x((B, C, H, W))

@pytest.fixture
def typical_bound_tokens_transformer():
    B=4
    W=64
    H=64
    C=96
    DET_TOKENS=100
    INTER_TOKENS=100
    return generate_x((B, H*W + DET_TOKENS+INTER_TOKENS, C))


def test_patch_split_dims_4x4_small_equal_divisible(typical_x_input):
    B=4
    W=64
    H=64
    PATCH_SIZE=4
    EMBED_DIM=48
    
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(typical_x_input).shape) == [B, EMBED_DIM, H/4, W/4]
    
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
    
def test_patch_split_dims_8x8(typical_x_input):
    B=4
    W=64
    H=64
    PATCH_SIZE=8
    EMBED_DIM=48
    
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(typical_x_input).shape) == [B, EMBED_DIM, H/PATCH_SIZE, W/PATCH_SIZE]
    
def test_patch_split_dims_1x1(typical_x_input):
    B=4
    W=64
    H=64
    PATCH_SIZE=1
    EMBED_DIM=48
    
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(typical_x_input).shape) == [B, EMBED_DIM, H/PATCH_SIZE, W/PATCH_SIZE]
    
def test_patch_split_0x0():
    PATCH_SIZE=0
    EMBED_DIM=48
    with pytest.raises(ValueError):
        create_patch_embed(PATCH_SIZE, EMBED_DIM)
        
def test_patch_split_dims_embed_large(typical_x_input):
    B=4
    W=64
    H=64
    PATCH_SIZE=4
    EMBED_DIM=96
    
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(typical_x_input).shape) == [B, EMBED_DIM, H/PATCH_SIZE, W/PATCH_SIZE]
    
def test_patch_split_dims_embed_xs(typical_x_input):
    B=4
    W=64
    H=64
    PATCH_SIZE=4
    EMBED_DIM=1
    
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM)
    assert list(patch_embed(typical_x_input).shape) == [B, EMBED_DIM, H/PATCH_SIZE, W/PATCH_SIZE]

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
    
def test_patch_split_weight_standardization(typical_x_input):
    B=4
    W=64
    H=64
    PATCH_SIZE=4
    EMBED_DIM=48
    
    patch_embed = create_patch_embed(PATCH_SIZE, EMBED_DIM, weight_standardization=True)
    assert list(patch_embed(typical_x_input).shape) == [B, EMBED_DIM, H/PATCH_SIZE, W/PATCH_SIZE]
    
def test_patch_merging(typical_bound_tokens_transformer):
    B=4
    W=64
    H=64
    IN_CHANNELS = 96
    DET_TOKENS=100
    INTER_TOKENS=100
    
    patch_merging = create_patch_merge(IN_CHANNELS, DET_TOKENS, INTER_TOKENS)
    out=patch_merging(typical_bound_tokens_transformer, H, W)
    assert list(out.shape) == [B, int(H/2 * W/2 + DET_TOKENS + INTER_TOKENS), IN_CHANNELS * 2]
    
def test_patch_merging_inequal_det_inter_tokens():
    B=4
    W=64
    H=64
    IN_CHANNELS = 96
    DET_TOKENS=100
    INTER_TOKENS=300
    
    patch_merging = create_patch_merge(IN_CHANNELS, DET_TOKENS, INTER_TOKENS)
    x = generate_x((B, H*W + DET_TOKENS + INTER_TOKENS, IN_CHANNELS))
    out=patch_merging(x, H, W)
    assert list(out.shape) == [B, int(H/2 * W/2 + DET_TOKENS + INTER_TOKENS), IN_CHANNELS * 2]
    
def test_patch_merging_indivisible():
    B=4
    W=49
    H=49
    IN_CHANNELS = 96
    DET_TOKENS=100
    INTER_TOKENS=100
    
    patch_merging = create_patch_merge(IN_CHANNELS, DET_TOKENS, INTER_TOKENS)
    x = generate_x((B, H*W + DET_TOKENS + INTER_TOKENS, IN_CHANNELS))
    out=patch_merging(x, H, W)
    assert list(out.shape) == [B, int(ceil(H/2) * ceil(W/2) + DET_TOKENS + INTER_TOKENS), IN_CHANNELS * 2]
    
def test_patch_merging_xs():
    B=4
    W=1
    H=1
    IN_CHANNELS = 96
    DET_TOKENS=100
    INTER_TOKENS=100
    
    patch_merging = create_patch_merge(IN_CHANNELS, DET_TOKENS, INTER_TOKENS)
    x = generate_x((B, H*W + DET_TOKENS + INTER_TOKENS, IN_CHANNELS))
    out=patch_merging(x, H, W)
    assert list(out.shape) == [B, int(ceil(H/2) * ceil(W/2) + DET_TOKENS + INTER_TOKENS), IN_CHANNELS * 2]
    
   
def test_patch_merging_expand(typical_bound_tokens_transformer):
    B=4
    W=64
    H=64
    IN_CHANNELS = 96
    DET_TOKENS=100
    INTER_TOKENS=100
    FIXED_OUT_CHANNELS=256
    patch_merging = create_patch_merge(IN_CHANNELS, DET_TOKENS, INTER_TOKENS, expand=False)
    # x = generate_x((B, H*W + DET_TOKENS + INTER_TOKENS, IN_CHANNELS))
    out=patch_merging(typical_bound_tokens_transformer, H, W)
    assert list(out.shape) == [B, int(H/2 * W/2 + DET_TOKENS + INTER_TOKENS), FIXED_OUT_CHANNELS]