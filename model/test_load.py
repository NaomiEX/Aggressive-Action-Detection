from inference import transform_img, load_trained_model, SimArgs, predict
from pathlib import Path
from PIL import Image
from util.misc import nested_tensor_from_tensor_list



import torchvision.transforms as T 

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

def test_img_load():
    filename=Path("model/data/surveillance/test/45.jpg")
    img=Image.open(filename).convert('RGB')
    transformed_img_tens, _ = transform_img(img)
    tensorToPIL = T.ToPILImage()
    transformed_img = tensorToPIL(transformed_img_tens)
    transformed_img.save("transformed_img_45.jpg")

def test_img_load_unnormalize():
    filename=Path("model/data/surveillance/test/45.jpg")
    img=Image.open(filename).convert('RGB')
    transformed_img_tens, _ = transform_img(img)
    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    tensorToPIL = T.ToPILImage()
    transformed_img = tensorToPIL(unorm(transformed_img_tens))
    transformed_img.save("transformed_img_45_unnorm.jpg")
    
def test_postprocess_preds():
    model = load_trained_model(Path("model/logs/surveillance/run_3/checkpoint.pth"), "cuda", SimArgs())
    filename=Path("model/data/surveillance/test/45.jpg")
    img=Image.open(filename).convert('RGB')
    w, h = img.size
    predict(model, "cuda", img, w, h, write_to_file=True, filename="test_postprocess_preds.json")
    
def test_postprocess_preds_no_fname():
    model = load_trained_model(Path("model/logs/surveillance/run_3/checkpoint.pth"), "cuda", SimArgs())
    filename=Path("model/data/surveillance/test/45.jpg")
    img=Image.open(filename).convert('RGB')
    w, h = img.size
    predict(model, "cuda", img, w, h, write_to_file=True)
    
def test_postprocess_preds_invalid_fname():
    model = load_trained_model(Path("model/logs/surveillance/run_3/checkpoint.pth"), "cuda", SimArgs())
    filename=Path("model/data/surveillance/test/45.jpg")
    img=Image.open(filename).convert('RGB')
    w, h = img.size
    try:
        predict(model, "cuda", img, w, h, write_to_file=True, filename="test_postprocess_preds.txt")
    except AssertionError as err:
        print("Correctly raised a ValueError with the following message: " + str(err))
    

if __name__ == "__main__":
    # test_img_load()
    # test_img_load_unnormalize()
    # test_postprocess_preds()
    # test_postprocess_preds_no_fname()
    test_postprocess_preds_invalid_fname()