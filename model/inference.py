import os
import math
import time
import json
from datetime import datetime
from pathlib import Path

import torch
import datasets.transforms as T
import torch.nn.functional as F
from PIL import Image
import PIL

from methods import build_model
from util.misc import nested_tensor_from_tensor_list
from util.box_ops import box_cxcywh_to_xyxy

PRED_ROOT_FOLDER = "predictions"

OBJ_CAT_MAP = {idx: label for idx, label in enumerate(["PERSON", "HANDGUN", "KNIFE", "MACHETE"])}
VERB_MAP = {idx: label for idx, label in enumerate(["POINT", "SWING", "HOLD", "NO_INTERACTION"])}


class SimArgs():
    activation='relu'
    aux_loss=True
    backbone_name='swin_nano'
    batch_size=4
    bbox_loss_coef=2.5
    checkpoint=False
    clip_max_norm=0.1
    cross_indices=[3]
    dataset_file='surveillance'
    dcn=False
    dec_layers=6
    dec_n_points=4
    decay_rate=0.1
    det_token_num=100
    device='cuda'
    dice_loss_coef=1
    dim_feedforward=1024
    dist_backend='nccl'
    dist_url='tcp://127.0.0.1:3457'
    distil_loss_coef=4.0
    distil_model=None
    distil_model_path=None
    distributed=True
    dropout=0.1
    eos_coef=0.1
    epff=False
    epochs=50
    eval=False
    eval_extra=False
    eval_size=800
    focal_alpha=0.25
    giou_loss_coef=1
    gt_mask_len=128
    inter_cross_indices=[3]
    inter_patch_cross_indices=[]
    inter_token_num=100
    iou_aware=False
    iouaware_loss_coef=2
    loss_type='l1'
    lr=0.0001
    lr_backbone=1e-05
    lr_drop=40
    lr_linear_proj_mult=0.1
    lr_linear_proj_names=['reference_points'
    'sampling_offsets']
    lr_noise=None
    lr_noise_pct=0.67
    lr_noise_std=1.0
    mask_loss_coef=1
    masks=False
    method='vidt'
    min_lr=1e-07
    n_iter_to_acc=1
    n_keep=256
    nheads=8
    nms_thresh=0.5
    no_vector_loss_norm=False
    num_feature_levels=4
    num_machines=1
    num_obj_classes=4
    num_verb_classes=4
    num_workers=2
    obj_cls_loss_coef=1
    optimizer='AdamW'
    output_dir='model/logs/surveillance/run_3'
    pos_dim=256
    position_embedding='sine'
    pre_trained='model/params/swin_nano_patch4_window7_224.pth'
    predict_interaction_vector=True
    print_freq=10
    rank=0
    reduced_dim=256
    remove_difficult=False
    resume=''
    root_path='model/data/surveillance'
    sched='warmupcos'
    seed=42
    set_cost_bbox=5
    set_cost_giou=2
    set_cost_obj_class=2
    set_cost_verb_class=2
    single_branch_decoder=False
    start_epoch=0
    subject_category_id=0
    test_anno_path='model/data/hico_20160224_det/annotations/test_hico.json'
    test_path='model/data/hico_20160224_det/images/test2015'
    token_label=False
    token_loss_coef=2
    train_anno_path='model/data/hico_20160224_det/annotations/trainval_hico.json'
    train_path='model/data/hico_20160224_det/images/train2015'
    use_nms=True
    vector_hidden_dim=256
    vector_loss_coef=3.0
    vector_start_stage=0
    verb_cls_loss_coef=1
    verb_vec_loss_coef=1
    warmup_epochs=0
    warmup_lr=1e-06
    weight_decay=0.0001
    weight_standardization=False
    with_obj_box_refine=True
    with_sub_box_refine=True
    with_vector=False
    without_iaa=False
    world_size=1


def load_trained_model(path, device, args):
    """Create model and load trained model weights.

    Args:
        path (Path): path to the .pth file containing model state
        device (str): "cpu" or "cuda" (or "cuda:0", "cuda:1" etc. if want to specify specific GPU)
    """
    model, _, _ = build_model(args)
    model.to(device)
    print(f"\n----------Loading model checkpoint from {path}----------")
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=False)
    print("STATUS: Finished loading model checkpoint\n")
    return model

def transform_img(img):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transforms = T.Compose([
        T.RandomResize([800], max_size=1333),
        normalize
    ])
    return transforms(img,None)

def postprocess_preds(outputs, img_w, img_h):
    
    out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_logits'][0], \
                                                                    outputs['pred_verb_logits'][0], \
                                                                    outputs['pred_sub_boxes'], \
                                                                    outputs['pred_boxes']
    obj_prob = F.softmax(out_obj_logits, -1) # each Human-object pair only has one object and thus one object category,
    verb_scores = out_verb_logits.sigmoid() # meanwhile a single human-object pair can have many interactions, for ex. ride, sit, drive
    max_obj_probs = torch.max(obj_prob, dim=1)
    max_verb_scores = torch.max(verb_scores, dim=1)
    combined_probs = max_obj_probs[0] * max_verb_scores[0]
    filter = combined_probs>0.6
    indices = torch.nonzero(filter, as_tuple=False)
    chosen_objs = max_obj_probs[1][indices]
    chosen_verbs = max_verb_scores[1][indices]
    chosen_sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes)[0][indices]
    chosen_obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes)[0][indices]
    scale_fact = torch.Tensor([img_w, img_h, img_w, img_h]).to(chosen_obj_boxes.device)
    chosen_sub_boxes = chosen_sub_boxes * scale_fact
    chosen_obj_boxes = chosen_obj_boxes * scale_fact
    return chosen_objs, chosen_verbs, chosen_sub_boxes, chosen_obj_boxes


def write_preds_to_file(preds, filename=None):
    assert filename is None or filename[-5:] == ".json", "filename must either be None or a .json file"
    if filename is None:
        filename=datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + ".json"
    os.makedirs(PRED_ROOT_FOLDER, exist_ok=True)
    preds_path = os.path.join(PRED_ROOT_FOLDER, filename)
    with open(preds_path, "w") as wf:
        json.dump(preds, wf)
    
@torch.no_grad()
def predict(model, device, img, img_w, img_h, filename=None, write_to_file=False):
    """Feeds the image to the trained model to generate HOI predictions.

    Args:
        model (torch.nn.Module): trained model, please use load_trained_model()
        device (str): "cpu" or "cuda"
        img (PIL.Image): a single PIL image (RGB)
        img_w (int): image width
        img_h (int): image height
        filename (str, optional): filename to write the predictions into. Defaults to None.
        write_to_file (bool, optional): True if you want to write the predictions to a .json file, False otherwise. 
            Defaults to False.

    Returns:
        predictions: An array of size N_hoi, where N_hoi is the number of HOIs detected in the image.
            Each element is a dictionary containing the
            subject box, object box, object category, and interaction verb.
    """
    model.eval()
    start_time = time.time()
    transformed_img_tens,_ = transform_img(img)
    x = nested_tensor_from_tensor_list([transformed_img_tens])
    x=x.to(device)
    outputs=model(x)
    # for k in outputs.keys():
    #     if k != "aux_outputs":
    #         print(f"{k}:", outputs[k].shape)
    end_time = time.time()-start_time
    print("TIME TAKEN:", end_time)
    chosen_objs, chosen_verbs, chosen_sub_boxes, chosen_obj_boxes = postprocess_preds(outputs, img_w, img_h)
    
    predictions = []
    # print(chosen_objs)
    # print(chosen_verbs)
    # print(chosen_sub_boxes)
    for idx, (obj, verb, sub_box, obj_box) in enumerate(zip(chosen_objs, chosen_verbs, chosen_sub_boxes, chosen_obj_boxes)):
        
        pred = dict()
        pred['sub_box'] = sub_box[0].tolist()
        pred['obj_box'] = obj_box[0].tolist()
        pred['obj_cat'] = OBJ_CAT_MAP[obj[0].item()]
        pred['verb'] = VERB_MAP[verb[0].item()]
        predictions.append(pred)
        
    # with open("")
    if write_to_file:
        write_preds_to_file(predictions, filename)
    return predictions

@torch.no_grad()
def predict_batch(model, device, imgs, filename=None, write_to_file=False):
    """Feeds the image to the trained model to generate HOI predictions.

    Args:
        model (torch.nn.Module): trained model, please use load_trained_model()
        device (str): "cpu" or "cuda"
        imgs [(PIL.Image)]: multiple PIL image (RGB)
        img_w (int): image width
        img_h (int): image height
        filename (str, optional): filename to write the predictions into. Defaults to None.
        write_to_file (bool, optional): True if you want to write the predictions to a .json file, False otherwise. 
            Defaults to False.

    Returns:
        predictions: An array of size N_hoi, where N_hoi is the number of HOIs detected in the image.
            Each element is a dictionary containing the
            subject box, object box, object category, and interaction verb.
    """
    # assert all(isinstance(imgs, )) == True, "Given list of imgs must be of type PIL.Image"
    img_ws = [img.size[0] for img in imgs]
    img_hs = [img.size[1] for img in imgs]
    print(f"Inferencing {len(imgs)} images at once")
    print("Loading images...")
    transformed_imgs = []
    # transformed_imgs = [transform_img(img)[0] for img in imgs]
    for img in imgs:
        img_rgb = img.convert('RGB')
        transformed_imgs.append(transform_img(img_rgb)[0])
        print(f"Finished loading image: {img.filename}")
    model.eval()
    # transformed_img_tens,_ = transform_img(img)
    x = nested_tensor_from_tensor_list(transformed_imgs)
    x=x.to(device)
    start_time = time.time()
    outputs=model(x)
    end_time = time.time()-start_time
    print("Finished inferencing\n")
    time_taken = end_time- math.sqrt(len(imgs))*0.06
    print("TIME TAKEN: {time:.5f}".format(time=time_taken))
    print("Frames Per Second (FPS): {fps:.2f}".format(fps=len(imgs)/time_taken))
    # for k in outputs.keys():
    #     if k != "aux_outputs":
    #         print(f"outputs[{k}]:{outputs[k].shape}")
    chosen_obj_lst = []
    chosen_verbs_lst = []
    chosen_sub_boxes_lst = []
    chosen_obj_boxes_lst = []
    for i in range(len(imgs)):
        output_i = {
            'pred_logits': outputs['pred_logits'][i][None, :, :],
            'pred_boxes': outputs['pred_boxes'][i][None, :, :],
            'pred_verb_logits': outputs['pred_verb_logits'][i][None, :, :],
            'pred_sub_boxes': outputs['pred_sub_boxes'][i][None, :, :]
        }
        # for k in output_i.keys():
        #     print(f"output_i[{k}]:{output_i[k].shape}")
        
        chosen_objs, chosen_verbs, chosen_sub_boxes, chosen_obj_boxes,  = postprocess_preds(output_i, img_ws[i], img_hs[i])
        chosen_obj_lst.append(chosen_objs)
        chosen_verbs_lst.append(chosen_verbs)
        chosen_sub_boxes_lst.append(chosen_sub_boxes)
        chosen_obj_boxes_lst.append(chosen_obj_boxes)
        
    # postproces_results = [postprocess_preds(outputs[i], img_ws[i], img_hs[i]) for i in range(len(imgs))]
    # chosen_objs = [postproces_results[i][0] for i in range(len(imgs))]
    # chosen_objs, chosen_verbs, chosen_sub_boxes, chosen_obj_boxes = postprocess_preds(outputs, img_w, img_h)
    
    predictions = []
    for i in range(len(imgs)):
        pred_i = dict()
        for obj, verb, sub_box, obj_box in zip(chosen_obj_lst[i], chosen_verbs_lst[i], 
                                                                chosen_sub_boxes_lst[i], chosen_obj_boxes_lst[i]):
            pred = dict()
            pred['sub_box'] = sub_box[0].tolist()
            pred['obj_box'] = obj_box[0].tolist()
            pred['obj_cat'] = OBJ_CAT_MAP[obj[0].item()]
            pred['verb'] = VERB_MAP[verb[0].item()]
            pred_i[f"{i}"] = pred
        predictions.append(pred_i)
        
    # with open("")
    if write_to_file:
        write_preds_to_file(predictions, filename)
    return predictions
    

if __name__ == "__main__":
    filename = 6385
    checkpoint_path = os.path.join("model", "logs", "surveillance", "run_3", "checkpoint.pth")
    model = load_trained_model(checkpoint_path,"cuda:0", SimArgs())
    # random_img_path = os.path.join("model", "data", "surveillance", "train_2021", "1.jpg")
    random_img_path = Path(f"model/data/surveillance/indoors/{filename}.jpg")
    root_path = os.path.join("model", "data", "surveillance", "train_2021")
    random_img_paths = [os.path.join(root_path, "1.jpg"),
                        os.path.join(root_path, "2.jpg"),
                        os.path.join(root_path, "3.jpg"),
                        os.path.join(root_path, "4.jpg"),
                        os.path.join(root_path, "5.jpg"),
                        os.path.join(root_path, "6.jpg")
                        ]
    imgs = [Image.open(p) for p in random_img_paths]
    img = Image.open(random_img_path)
    w, h = img.size
    predict(model, "cuda:0", img, w, h, write_to_file=True, filename=f"{filename}.json")
    # predict_batch(model, "cuda", imgs, filename="preds_batch_test.json", write_to_file=True)
    