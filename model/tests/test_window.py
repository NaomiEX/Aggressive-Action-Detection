import pytest
from test_helper_fns import generate_x
from methods.swin_w_ram import window_partition, window_reverse


def test_window_partition():
    B = 4
    H = 64
    W = 64
    C = 96
    WINDOW_SIZE=8
    x = generate_x((B, H, W, C))
    out = window_partition(x, WINDOW_SIZE)
    num_windows = H // WINDOW_SIZE * W // WINDOW_SIZE
    assert list(out.shape) == [B * num_windows, WINDOW_SIZE, WINDOW_SIZE, C]
    
def test_window_partition_large_window():
    B = 4
    H = 64
    W = 64
    C = 96
    WINDOW_SIZE=80
    x = generate_x((B, H, W, C))
    with pytest.raises(ValueError):
        window_partition(x, WINDOW_SIZE)
        
def test_window_partition_xs_window():
    B = 4
    H = 64
    W = 64
    C = 96
    WINDOW_SIZE=1
    x = generate_x((B, H, W, C))
    out = window_partition(x, WINDOW_SIZE)
    num_windows = H // WINDOW_SIZE * W // WINDOW_SIZE
    assert list(out.shape) == [B * num_windows, WINDOW_SIZE, WINDOW_SIZE, C]
    
def test_window_reverse():
    B = 4
    H = 64
    W = 64
    C = 96
    WINDOW_SIZE=8
    x = generate_x((B, H, W, C))
    interm = window_partition(x, WINDOW_SIZE)
    out=window_reverse(interm, WINDOW_SIZE, H, W)
    # num_windows = H // WINDOW_SIZE * W // WINDOW_SIZE
    assert list(out.shape) == [B, H, W, C]
    
def test_window_reverse_mismatch_window():
    B = 4
    H = 64
    W = 64
    C = 96
    WINDOW_SIZE=8
    x = generate_x((B, H, W, C))
    interm = window_partition(x, WINDOW_SIZE)
    interm = interm.repeat((1, 2, 2, 1))
    with pytest.raises(ValueError):
        window_reverse(interm, WINDOW_SIZE, H, W)
    