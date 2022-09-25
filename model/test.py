
from datasets.surveillance import build

if __name__ == "__main__":
    class args():
        root_path = "data/surveillance"
        det_token_num = 100
        inter_token_num = 100
    ds = build("train", args())
    
    ds_test = build("val", args())
    
    x = ds[0]