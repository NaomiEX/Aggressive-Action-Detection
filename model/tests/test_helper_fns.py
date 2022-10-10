import torch
import numpy as np
SEED = 42

def generate_x(size, *args, **kwargs):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return torch.rand(size, *args, **kwargs)

def generate_mask_all1(size, *args, **kwargs):
    return torch.ones(size, *args, **kwargs)

def generate_mask_all0(size, *args, **kwargs):
    return torch.zeros(size, *args, **kwargs)