import torch
import numpy as np
SEED = 42

def generate_x(size, *args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return torch.rand(size, *args)