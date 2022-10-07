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

class SurveillanceDataset(torch.utils.data.Dataset):
    def __init__(self, img_set, img_folder, anno_file, transforms, 
                 num_queries, num_obj_classes=4, num_verbs=4):
        """Load all the annotations and get valid object ids and valid verbs

        Args:
            img_set (str): "train" or "val" set
            img_folder (Path): path to root image folder
            anno_file (Path): path to annotation file
            transforms (torch.transforms): composed transforms for the images and annotations
            num_queries (int): number of queries in the decoder, determines max. number of interactions that can be detected
            num_obj_classes (int, optional): number of possible object classes, assumes 1-indexed. Defaults to 4.
            num_verbs (int, optional): number of possible verbs, assume 1-indexed. Defaults to 4.

        Returns:
            torch.utils.data.Dataset: dataset object for the Surveillance Dataset
        """
        self.img_set = img_set
        self.img_folder = img_folder
        
        # load the annotations, an example annotation is given below:
        # {
            # "file_name": "0.jpg", 
            # "img_id": 1, 
            # "annotations": [
                # {"bbox": ["620", "89", "1018", "449"], "category_id": 1}, 
                # {"bbox": ["649", "224", "704", "276"], "category_id": 2}, 
                # {"bbox": ["650", "204", "1082", "554"], "category_id": 1}], 
            # "hoi_annotation": [
                # {"subject_id": 0, "object_id": 1, "category_id": 1, "hoi_category_id": 2}, 
                # {"subject_id": 2, "object_id": 1, "category_id": 4, "hoi_category_id": 3}]
        # },
        # where category_id in "annotations" refer to the object category, 
        # whereas category_id in hoi_annotation refer to the verb.
        # "hoi_category_id" is the object-verb pair, for ex. "handgun point" or "knife swing"
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        self._transforms = transforms

        self.num_queries = num_queries
        
        # possible object categories (1-indexed)
        self._valid_obj_ids = list(range(1, num_obj_classes+1))
        # possible  interaction categories (1-indexed)
        self._valid_verb_ids = list(range(1, num_verbs+1))

        if img_set == 'train':
            self.ids = [] # list of indices to valid image annotations 
            for idx, img_anno in enumerate(self.annotations):
                for hoi in img_anno['hoi_annotation']:
                    if hoi['subject_id'] >= len(img_anno['annotations']) or hoi['object_id'] >= len(img_anno['annotations']):
                        break
                else: # if none of the image's hoi annotations are invalid append it to self.ids
                    self.ids.append(idx)
        else:
            self.ids = list(range(len(self.annotations)))
            
    def __len__(self):
        """Get total number of data in the dataset

        Returns:
            int: number of images in the dataset
        """
        return len(self.ids)
    
    def __getitem__(self, idx):
        # get only from the list of valid images, so index the ids list
        # this gives a valid image index, then use it to retrieve the corresponding annotations
        img_anno = self.annotations[self.ids[idx]]

        # open the relevant image and convert to RGB
        img = Image.open(self.img_folder / img_anno['file_name']).convert('RGB')
        w, h = img.size

        # if there are more object annotations than number of queries, limit it (very unlikely to trigger)
        if self.img_set == 'train' and len(img_anno['annotations']) > self.num_queries:
            img_anno['annotations'] = img_anno['annotations'][:self.num_queries]

        # get the 4-bbox coordinates for every single object in the image
        boxes = [[int(bbox_coord) for bbox_coord in obj['bbox']] for obj in img_anno['annotations']]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4) # t.s.: [N_o, 4]
        
        if self.img_set == 'train':
            # Add index for confirming which boxes are kept after image transformation
            # store the object classes as the index of the _valid_obj_ids list
            # this is more convenient because object categories are 1-indexed and by storing the index to the list instead,
            # we can convert it to be 0-indexed
            classes = [(i, self._valid_obj_ids.index(obj['category_id'])) for i, obj in enumerate(img_anno['annotations'])]
        else:
            classes = [self._valid_obj_ids.index(obj['category_id']) for obj in img_anno['annotations']]
        classes = torch.tensor(classes, dtype=torch.int64)
        
        target = {}
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        target['size'] = torch.as_tensor([int(h), int(w)])
        if self.img_set == 'train':
            # ensure that the annotations are not out-of-bounds
            boxes[:, 0::2].clamp_(min=0, max=w)
            boxes[:, 1::2].clamp_(min=0, max=h)
            
            # only keep bboxed for which x2>x1 and y2>y1
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            classes = classes[keep]

            target['boxes'] = boxes
            target['labels'] = classes
            # target['iscrowd'] = torch.tensor([0 for _ in range(boxes.shape[0])])
            target['area'] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

            if self._transforms is not None:
                img, target = self._transforms(img, target)

            # indices of object annotations that were kept
            kept_box_indices = [label[0] for label in target['labels']]

            # only keep the class labels
            target['labels'] = target['labels'][:, 1]

            obj_labels, verb_labels, sub_boxes, obj_boxes = [], [], [], []
            sub_obj_pairs = []
            # verb_label_enc = [0 for _ in range(len(self._valid_verb_ids))]
            verb_vecs = []
            for hoi in img_anno['hoi_annotation']:
                if hoi['subject_id'] not in kept_box_indices or hoi['object_id'] not in kept_box_indices:
                    continue
                sub_obj_pair = (hoi['subject_id'], hoi['object_id'])
                # verb_label_enc[self._valid_verb_ids.index(hoi['category_id'])] = 1
                if sub_obj_pair in sub_obj_pairs:
                    verb_labels[sub_obj_pairs.index(sub_obj_pair)][self._valid_verb_ids.index(hoi['category_id'])] = 1
                else:
                    sub_obj_pairs.append(sub_obj_pair)
                    obj_labels.append(target['labels'][kept_box_indices.index(hoi['object_id'])])
                    verb_label = [0 for _ in range(len(self._valid_verb_ids))]
                    verb_label[self._valid_verb_ids.index(hoi['category_id'])] = 1
                    sub_box = target['boxes'][kept_box_indices.index(hoi['subject_id'])]
                    obj_box = target['boxes'][kept_box_indices.index(hoi['object_id'])]
                    
                    sub_ct = sub_box[..., :2] # cx, cy normalized
                    obj_ct = obj_box[..., :2] # cx, cy normalized
                    
                    verb_vec = torch.cat([sub_ct, obj_ct], dim=-1) # cx_sub, cy_sub, cx_obj, cy_obj

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
        else:
            target['boxes'] = boxes
            target['labels'] = classes
            target['id'] = idx
            target['file_name'] = img_anno['file_name']

            if self._transforms is not None:
                img, _ = self._transforms(img, None)

            hois = []
            for hoi in img_anno['hoi_annotation']:
                hois.append((hoi['subject_id'], hoi['object_id'], self._valid_verb_ids.index(hoi['category_id'])))
            target['hois'] = torch.as_tensor(hois, dtype=torch.int64)

        return img, target
        
# Add color jitter to coco transforms
def make_hico_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.ColorJitter(.4, .4, .4),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
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
    # root = Path(args.hoi_path)
    root = Path(args.root_path)
    assert root.exists(), f'provided HOI path {root} does not exist'
    PATHS = {
        'train': (root / 'train_2021', root / 'train_2021.json'),
        'val': (root / 'test', root / 'test_anno.json')
    }
    # CORRECT_MAT_PATH = root / 'annotations' / 'corre_hico.npy'

    img_folder, anno_file = PATHS[image_set]
    dataset = SurveillanceDataset(image_set, img_folder, anno_file, make_hico_transforms(image_set),
                            args.det_token_num)
    # if image_set == 'val':
        # dataset.set_rare_hois(PATHS['train'][1])
        # dataset.load_correct_mat(CORRECT_MAT_PATH)
    return dataset

if __name__ == "__main__":
    class args():
        root_path = "data/surveillance"
        det_token_num = 100
        inter_token_num = 100
    build("train", args())