import torch
from torch.nn import LayerNorm
# from datasets.surveillance import build
from methods.swin_w_ram import SwinTransformer, PatchEmbed


if __name__ == "__main__":
    # class args():
    #     root_path = "data/surveillance"
    #     det_token_num = 100
    #     inter_token_num = 100
    # ds = build("train", args())
    
    # ds_test = build("val", args())
    
    # x = ds[0]
    
    p=PatchEmbed(patch_size=4, embed_dim=48, norm_layer=LayerNorm)
    x = torch.rand((4, 3, 0, 0))
    print(list(p(x).shape))