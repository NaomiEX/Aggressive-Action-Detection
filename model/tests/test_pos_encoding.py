import pytest
import torch
import math

from methods.swin_w_ram import masked_sin_pos_encoding
from test_helper_fns import generate_x

@pytest.fixture
def typical_pat_tokens():
    B=4
    H=64
    W=64
    C=96
    return generate_x((B, H, W, C))

@pytest.fixture
def typical_pat_mask():
    B=4
    H=64
    W=64
    return torch.ones((B, H, W)).bool()

def test_masked_sin_pos_encoding(typical_pat_tokens, typical_pat_mask):
    B=4
    H=64
    W=64
    NUM_POS_FEATS = 96
    out = masked_sin_pos_encoding(typical_pat_tokens, typical_pat_mask, NUM_POS_FEATS)
    assert list(out.shape) == [B, H, W, NUM_POS_FEATS]
    
def test_masked_sin_pos_encoding_diff_pos_feats(typical_pat_tokens, typical_pat_mask):
    B=4
    H=64
    W=64
    NUM_POS_FEATS = 256
    out = masked_sin_pos_encoding(typical_pat_tokens, typical_pat_mask, NUM_POS_FEATS)
    assert list(out.shape) == [B, H, W, NUM_POS_FEATS]
    
def test_masked_sin_pos_encoding_diff_temp(typical_pat_tokens, typical_pat_mask):
    B=4
    H=64
    W=64
    NUM_POS_FEATS = 256
    TEMPERATURE=500
    out = masked_sin_pos_encoding(typical_pat_tokens, typical_pat_mask, NUM_POS_FEATS,
                                  temperature=TEMPERATURE)
    assert list(out.shape) == [B, H, W, NUM_POS_FEATS]
    
def test_masked_sin_pos_encoding_diff_scale(typical_pat_tokens, typical_pat_mask):
    B=4
    H=64
    W=64
    NUM_POS_FEATS = 256
    SCALE = math.pi
    
    out = masked_sin_pos_encoding(typical_pat_tokens, typical_pat_mask, NUM_POS_FEATS,
                                  scale=SCALE)
    assert list(out.shape) == [B, H, W, NUM_POS_FEATS]