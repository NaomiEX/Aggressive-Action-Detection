# ------------------------------------------------------------------------
# QAHOI
# Copyright (c) 2021 Junwen Chen. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from QPIC (https://github.com/hitachi-rd-cv/qpic)
# Copyright (c) Hitachi, Ltd. All Rights Reserved.
# ------------------------------------------------------------------------


"""
HOIA detection dataset.
"""
import os
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import numpy as np

import torch
import torch.utils.data
import torchvision
import math

import datasets.transforms as T
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


class HOIADataset(torch.utils.data.Dataset):

    def __init__(self, img_set, img_folder, anno_file, transforms, num_queries, num_inter_queries, args):
        """ Initializes HOIA Dataset object
        
        Args:
            img_set (str): 'train' or 'val'
            img_folder (str): path to images
            anno_file (str): path to annotation
            transforms (torchvision.transforms.transforms.Compose): transformations to be performed on the data
            num_queries (int): number of object queries in the transformer
            num_inter_queries (int): number of interaction queries in the transformer
        """
        self.img_set = img_set
        self.img_folder = img_folder
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries
        self.num_inter_queries = num_inter_queries

        self._valid_obj_ids = list(range(1, args.num_obj_classes + 1 )) # for hoia: [1,11]
        self._valid_verb_ids = list(range(1, args.num_verb_classes + 1)) # for hoia: [1,10]

        if img_set == 'train':
            self.ids = []
            for idx, img_anno in enumerate(self.annotations):
                flag_bad = False
                
                if len(img_anno['annotations']) > self.num_queries or len(img_anno['hoi_annotation']) > self.num_inter_queries:
                    flag_bad = True
                    continue
                for obj in img_anno["annotations"]:
                    if int(obj["category_id"]) not in self._valid_obj_ids:
                        flag_bad=True
                        break
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(img_anno['annotations']) or hoi["category_id"] not in self._valid_verb_ids:
                        flag_bad = True
                        break
                if not flag_bad:
                    self.ids.append(idx)


        else:
                self.ids = list(range(len(self.annotations)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        valid_img_id = self.ids[idx]
        img_anno = self.annotations[valid_img_id]
        # print("img_filename:", img_anno["file_name"])
        img_path = os.path.join(self.img_folder, img_anno['file_name'])
        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        # if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
        #     img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        # 2d list of size: N_o, 4
        boxes = [obj['bbox'] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4) # shape: [N_o, 4], where N_o is the number of objects in the image

        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            # List([obj_anno_id, valid_obj_category_id]), 
            # where valid_obj_category_id is the index in self._valid_obj_ids which the object belongs to
            classes = [(i, self._valid_obj_ids.index(int(obj['category_id']))) for i, obj in enumerate(img_anno['annotations'])]
        else:
            # print("img_filename:", img_anno['file_name'])
            classes = [self._valid_obj_ids.index(int(obj['category_id'])) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64) # shape: [N_o, 2]

        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        
        if self.img_set == 'train':
            # HOIA follows x_1, y_1, x_2, y_2 scheme
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            # ensures that x_2 > x_1 and y_2 > y_1
            # in HOIA all annotations satisfy this
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes # shape [N_o, 2]
            # # all 0s, N/A
            # target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            # (x_2 - x_1) * (y_2 - y_1) = width * height
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img, target = self._transforms(img, target) # boxes are now cxcywh normalized by img width and height

            kept_box_indices = [label[0] for label in target['labels']]

            # get the valid_obj_category_id for every label
            target['labels'] = target['labels'][:, 1] # [N_o]

            # obj_labels is the list of categories involved in the hois in order [N_iu], 
            #   where N_iu is the number of HOIs involving unique (subject, object) pairs
            # verb_labels is the list of categories of interactions of the hois in order [N_iu, 11], where 11 is the number of valid obj_classes
            #   NOTE: these verb_labels may not be one-hot encoded because if there is >1 interaction between the same sub-obj pair, 
            #         then there can be multiple 1s in the vector
            # sub_boxes is the list of bboxes of the subjects (humans) involved in hois in order [N_iu, 4]
            # obj_boxes is the list of bboxes of the objects involved in hois in order [N_iu, 4]
            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            # list of tuples containing the index of the subject and the index of the object [N_iu, 4]
            sub_obj_pairs = []
            
            verb_vecs = []
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                ## both subject and object bboxes are legal
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                
                if sub_obj_pair in sub_obj_pairs: # if there are multiple interactions involving the same subject and object,
                    # get the index where the previous pair was stored, then place a 1 in the relevant interaction category
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    # get the label for the object involved in the HOI
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1 # one-hot encoding of the interaction category
                    
                    ## gets the subject and object bboxes
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    
                    ## get the centerpoints of the subject and object in the HOI
                    sub_ct = sub_box[..., :2] # cx, cy normalized
                    obj_ct = obj_box[..., :2] # cx, cy normalized
                    verb_vec = torch.cat([sub_ct, obj_ct], dim=-1)
                    verb_labels.append(verb_label)
                    sub_boxes.append(sub_box)
                    obj_boxes.append(obj_box)
                    verb_vecs.append(verb_vec)

            if len(sub_obj_pairs) == 0:
                target['obj_labels'] = torch.zeros((0,), dtype=torch.int64)
                target['verb_labels'] = torch.zeros((0, len(self._valid_verb_ids)), dtype=torch.float32)
                target['sub_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['obj_boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                target['verb_vecs'] = torch.zeros((0, 4), dtype=torch.float32)
                # target['verb_label_enc'] = torch.zeros(len(self._valid_verb_ids), dtype=torch.float32)
            else:
                target['obj_labels'] = torch.stack(obj_labels)
                target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
                target['sub_boxes'] = torch.stack(sub_boxes)
                target['obj_boxes'] = torch.stack(obj_boxes)
                target['verb_vecs'] = torch.stack(verb_vecs)
                # target['verb_label_enc'] = torch.as_tensor(verb_label_enc, dtype=torch.float32)
            target['file_name'] = img_anno['file_name']
        else: # if image_set == 'val'
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx
            target['file_name'] = img_anno['file_name']
            

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            hois = []
            # obj_labels = []
            # verb_labels = []
            # sub_boxes = []
            # obj_boxes = []
            # verb_vecs = []
            
            # boxes = target['boxes']
            # labels = target['labels']
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
                # obj_bbox = boxes[hoi['object_id']]
                # sub_bbox = boxes[hoi['subject_id']]
                # obj_boxes.append(obj_bbox)
                # sub_boxes.append(sub_bbox)
                
                # obj_labels.append(labels[hoi['object_id']])
                # verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                # verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1 # one-hot encoding of the interaction category
                # verb_labels.append(verb_label)
                
                # obj_ct = obj_bbox[..., :2]
                # sub_ct = sub_bbox[..., :2]
                
                # verb_vec = torch.cat([sub_ct, obj_ct], dim=-1)
                # verb_vecs.append(verb_vec)
                
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)
            # target['obj_labels'] = torch.stack(obj_labels)
            # target['verb_labels'] = torch.as_tensor(verb_labels, dtype=torch.float32)
            # target['sub_boxes'] = torch.stack(sub_boxes)
            # target['obj_boxes'] = torch.stack(obj_boxes)
            # target['verb_vecs'] = torch.stack(verb_vecs)
        # print("target:", target)
        return img, target

    # def set_rare_hois(self, anno_file):
    #     with open(anno_file, 'r') as f:
    #         annotations = json.load(f)

    #     counts = defaultdict(lambda: 0)
    #     for img_id, img_anno in enumerate(annotations):
    #         if len(img_anno['annotations']) > self.num_queries:
    #             continue
    #         hois = img_anno['hoi_annotation']
    #         bboxes = img_anno['annotations']
    #         for obj in bboxes:
    #             if int(obj["category_id"]) not in self._valid_obj_ids:
    #                 flag_bad=True
    #                 break
    #         for hoi in hois:
    #             flag_bad = False
    #             for hoi in img_anno['hoi_annotation']:
                    
    #                 if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(img_anno['annotations']) or hoi["category_id"] not in self._valid_verb_ids:
    #                     flag_bad = True
    #                     break
    #             if flag_bad:
    #                 break
    #             triplet = (self._valid_obj_ids.index(int(bboxes[hoi['subject_id']]['category_id'])),
    #                         self._valid_obj_ids.index(int(bboxes[hoi['object_id']]['category_id'])),
    #                         self._valid_verb_ids.index(int(hoi['category_id'])))
    #             counts[triplet] += 1
    #     self.rare_triplets = []
    #     self.non_rare_triplets = []
    #     for triplet, count in counts.items():
    #         if count < 10:
    #             self.rare_triplets.append(triplet)
    #         else:
    #             self.non_rare_triplets.append(triplet)

    def load_correct_mat(self, path):
        self.correct_mat = np.load(path)


# Add color jitter to coco transforms
def make_hoia_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        # here is also where the target is converted to cx, cy, w, h
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            # brightness_factor, contrast_factor, saturation_factor range = [0.6, 1.4]
            # hue = 0 (remains unchanged)
            T.ColorJitter(.4, .4, .4),
            # randomly select between the two transformations with 0.5 probability
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]), # no max size (unbounded resize)
                    # randomly crop it to a boc between 384x384 to 600x600 (provided 600 <= image width and height)
                    T.RandomSizeCrop(384, 600),
                    # Randomly resize the image again but this time following the normal scales
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.root_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (args.train_path, args.train_anno_path),
        'val': (args.test_path, args.test_anno_path)
    }
    CORRECT_MAT_PATH = os.path.join(root, 'annotations', 'corre_hoia.npy')

    img_folder, anno_file = PATHS[image_set]
    dataset = HOIADataset(image_set, img_folder, anno_file, transforms=make_hoia_transforms(image_set),
                            num_queries=args.det_token_num, num_inter_queries=args.inter_token_num, args=args)
    if image_set == 'val':
        # dataset.set_rare_hois(PATHS['train'][1])
        dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset

if __name__ == "__main__":
    class args():
        root_path = "/data/hoia"
        train_path = "/data/hoia/images/train_nano"
        train_anno_path = "/data/hoia/annotations/train_nano_anno.json"
        test_path = "/data/hoia/images/test_nano"
        test_anno_path = "/data/hoia/annotations/test_nano_anno.json"
        det_token_num = 100
        inter_token_num = 100
        num_obj_classes = 11
        num_verb_classes = 10
    build("train", args())