# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision
from .coco import build as build_coco
from .voc import build as build_voc

from .hoia import build as build_hoia
from .hico_det import build as build_hico_det
from .surveillance import build as build_surveillance

def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, args):

    # if args.dataset_file == 'coco':
    #     return build_coco(image_set, args)

    # if args.dataset_file == 'voc':
    #     return build_voc(image_set, args)
    
    if args.dataset_file == 'hoia':
        return build_hoia(image_set, args)
    elif args.dataset_file == 'hico-det':
        # print("UNDER CONSTRUCTION")
        return build_hico_det(image_set, args)
    elif args.dataset_file == 'vcoco':
        raise NotImplementedError("VCOCO dataset is coming soon")
    elif args.dataset_file == 'surveillance':
        return build_surveillance(image_set, args)

    raise ValueError(f'dataset {args.dataset_file} not supported')
