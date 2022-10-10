import pytest
from torch import nn

from methods.swin_w_ram import Mlp
from test_helper_fns import generate_x

def test_mlp_dims_larger_hidden_larger_output():
    B=4
    H=64
    W=64
    NUM_IN_FEATURES=48
    NUM_HIDDEN_FEATURES=96
    NUM_OUTPUT_FEATURES = 192
    x=generate_x((B, H, W, NUM_IN_FEATURES))
    mlp = Mlp(in_features=NUM_IN_FEATURES, hidden_features=NUM_HIDDEN_FEATURES, out_features=NUM_OUTPUT_FEATURES)
    assert list(mlp(x).shape) == [B, H, W, NUM_OUTPUT_FEATURES]
    
def test_mlp_dims_smaller_hidden_larger_output():
    B=4
    H=64
    W=64
    NUM_IN_FEATURES=48
    NUM_HIDDEN_FEATURES=24
    NUM_OUTPUT_FEATURES = 192
    x=generate_x((B, H, W, NUM_IN_FEATURES))
    mlp = Mlp(in_features=NUM_IN_FEATURES, hidden_features=NUM_HIDDEN_FEATURES, out_features=NUM_OUTPUT_FEATURES)
    assert list(mlp(x).shape) == [B, H, W, NUM_OUTPUT_FEATURES]
    
def test_mlp_dims_larger_hidden_smaller_output():
    B=4
    H=64
    W=64
    NUM_IN_FEATURES=48
    NUM_HIDDEN_FEATURES=96
    NUM_OUTPUT_FEATURES = 12
    x=generate_x((B, H, W, NUM_IN_FEATURES))
    mlp = Mlp(in_features=NUM_IN_FEATURES, hidden_features=NUM_HIDDEN_FEATURES, out_features=NUM_OUTPUT_FEATURES)
    assert list(mlp(x).shape) == [B, H, W, NUM_OUTPUT_FEATURES]

def test_mlp_mismatch_in_features():
    B=4
    C=96
    H=64
    W=64
    NUM_IN_FEATURES=48
    NUM_HIDDEN_FEATURES=96
    NUM_OUTPUT_FEATURES = 192
    x=generate_x((B, H, W, C))
    mlp = Mlp(in_features=NUM_IN_FEATURES, hidden_features=NUM_HIDDEN_FEATURES, out_features=NUM_OUTPUT_FEATURES)
    with pytest.raises(ValueError):
        mlp(x)
        
def test_mlp_dims_no_hidden():
    B=4
    H=64
    W=64
    NUM_IN_FEATURES=48
    NUM_HIDDEN_FEATURES=None
    NUM_OUTPUT_FEATURES = 12
    x=generate_x((B, H, W, NUM_IN_FEATURES))
    mlp = Mlp(in_features=NUM_IN_FEATURES, hidden_features=NUM_HIDDEN_FEATURES, out_features=NUM_OUTPUT_FEATURES)
    assert list(mlp(x).shape) == [B, H, W, NUM_OUTPUT_FEATURES]
    
def test_mlp_dims_no_output():
    B=4
    H=64
    W=64
    NUM_IN_FEATURES=48
    NUM_HIDDEN_FEATURES=96
    NUM_OUTPUT_FEATURES = None
    x=generate_x((B, H, W, NUM_IN_FEATURES))
    mlp = Mlp(in_features=NUM_IN_FEATURES, hidden_features=NUM_HIDDEN_FEATURES, out_features=NUM_OUTPUT_FEATURES)
    assert list(mlp(x).shape) == [B, H, W, NUM_IN_FEATURES]
    
def test_mlp_dims_no_hidden_no_output():
    B=4
    H=64
    W=64
    NUM_IN_FEATURES=48
    NUM_HIDDEN_FEATURES=None
    NUM_OUTPUT_FEATURES = None
    x=generate_x((B, H, W, NUM_IN_FEATURES))
    mlp = Mlp(in_features=NUM_IN_FEATURES, hidden_features=NUM_HIDDEN_FEATURES, out_features=NUM_OUTPUT_FEATURES)
    assert list(mlp(x).shape) == [B, H, W, NUM_IN_FEATURES]
    
def test_mlp_dims_activation_relu():
    B=4
    H=64
    W=64
    NUM_IN_FEATURES=48
    NUM_HIDDEN_FEATURES=96
    NUM_OUTPUT_FEATURES = 12
    ACTIVATION=nn.ReLU
    x=generate_x((B, H, W, NUM_IN_FEATURES))
    mlp = Mlp(in_features=NUM_IN_FEATURES, hidden_features=NUM_HIDDEN_FEATURES, out_features=NUM_OUTPUT_FEATURES, act_layer=ACTIVATION)
    assert list(mlp(x).shape) == [B, H, W, NUM_OUTPUT_FEATURES]
    
def test_mlp_dims_activation_tanh():
    B=4
    H=64
    W=64
    NUM_IN_FEATURES=48
    NUM_HIDDEN_FEATURES=96
    NUM_OUTPUT_FEATURES = 12
    ACTIVATION=nn.Tanh
    x=generate_x((B, H, W, NUM_IN_FEATURES))
    mlp = Mlp(in_features=NUM_IN_FEATURES, hidden_features=NUM_HIDDEN_FEATURES, out_features=NUM_OUTPUT_FEATURES, act_layer=ACTIVATION)
    assert list(mlp(x).shape) == [B, H, W, NUM_OUTPUT_FEATURES]
    
def test_mlp_dims_dropout_half():
    B=4
    H=64
    W=64
    NUM_IN_FEATURES=48
    NUM_HIDDEN_FEATURES=96
    NUM_OUTPUT_FEATURES = 12
    DROPOUT_RATE=0.5
    x=generate_x((B, H, W, NUM_IN_FEATURES))
    mlp = Mlp(in_features=NUM_IN_FEATURES, hidden_features=NUM_HIDDEN_FEATURES, out_features=NUM_OUTPUT_FEATURES, drop=DROPOUT_RATE)
    assert list(mlp(x).shape) == [B, H, W, NUM_OUTPUT_FEATURES]
    
def test_mlp_dims_dropout_all():
    B=4
    H=64
    W=64
    NUM_IN_FEATURES=48
    NUM_HIDDEN_FEATURES=96
    NUM_OUTPUT_FEATURES = 12
    DROPOUT_RATE=1.0
    x=generate_x((B, H, W, NUM_IN_FEATURES))
    mlp = Mlp(in_features=NUM_IN_FEATURES, hidden_features=NUM_HIDDEN_FEATURES, out_features=NUM_OUTPUT_FEATURES, drop=DROPOUT_RATE)
    assert list(mlp(x).shape) == [B, H, W, NUM_OUTPUT_FEATURES]
    assert (mlp(x)==0).all().item() == True