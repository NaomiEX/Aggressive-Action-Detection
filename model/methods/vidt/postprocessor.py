# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import torch
from torch import nn
from util import box_ops
import torch.nn.functional as F
from util.detectron2.utils.memory import retry_if_cuda_oom
from util.detectron2.layers.mask_ops import paste_masks_in_image
from util.box_ops import box_cxcywh_to_xyxy
import numpy as np
import cv2

class PostProcessHOI(nn.Module):
  def __init__(self, subject_category_id) -> None:
    super().__init__()
    self.subject_category_id = subject_category_id
    
  @torch.no_grad()
  def forward(self, outputs, target_sizes):
    out_obj_logits, out_verb_logits, out_sub_boxes, out_obj_boxes = outputs['pred_logits'], \
                                                                    outputs['pred_verb_logits'], \
                                                                    outputs['pred_sub_boxes'], \
                                                                    outputs['pred_boxes']
    
    obj_prob = F.softmax(out_obj_logits, -1) # each Human-object pair only has one object and thus one object category,
    verb_scores = out_verb_logits.sigmoid() # meanwhile a single human-object pair can have many interactions, for ex. ride, sit, drive
    
    num_verb_classes = verb_scores.shape[-1]
    num_obj_classes = obj_prob.shape[-1]
    
    topk_values, topk_indexes = torch.topk(obj_prob.flatten(1), 100, dim=1) # (torch.Size([B, 100]), torch.Size([B, 100]))
    obj_scores = topk_values
    
    # gets the query which is responsible for the top_i score
    topk_boxes = topk_indexes // num_obj_classes # shape: [B, 100]
    # since the queries are meant to be probabilities for repeating classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 1, 2, 3, ...
    # by doing % we get the label, for ex. index 12 represents probability of 1, 12 % 11 = 1 
    obj_labels = topk_indexes % num_obj_classes
    
    verb_scores = torch.gather(verb_scores, 1, topk_boxes.unsqueeze(-1).repeat(1,1,num_verb_classes)) # shape: [B, 100, num_verb_classes]
    out_obj_boxes = torch.gather(out_obj_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
    out_sub_boxes = torch.gather(out_sub_boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4))
    
    img_h, img_w = target_sizes.unbind(1)
    
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(verb_scores.device) # shape: [B, 4]
    sub_boxes = box_cxcywh_to_xyxy(out_sub_boxes) # shape: [B, 100, 4]
    obj_boxes = box_cxcywh_to_xyxy(out_obj_boxes) # shape: [B, 100, 4]
    
    sub_boxes = sub_boxes * scale_fct[:, None, :] # shape: [B, 100, 4]
    obj_boxes = obj_boxes * scale_fct[:, None, :] # shape: [B, 100, 4]
    
    results = []
    for os, ol, vs, sb, ob in zip(obj_scores, obj_labels, verb_scores, sub_boxes, obj_boxes):
      # os shape: [100], ol shape: [100], vs shape [100, 10], sb shape [100, 4], ob shape [100, 4]
      
      # 100-dimensional tensor full of 1s
      sl = torch.full_like(ol, self.subject_category_id) # shape: [100]
      l = torch.cat((sl, ol)) # shape: [200]
      b = torch.cat((sb, ob)) # shape: [200, 4]
      results.append({'labels': l.to('cpu'), 'boxes': b.to('cpu')})

      # "os.unsqueeze(1)" -> tensor of shape [100, 1]
      # verb sigmoid score is multiplied by the probability of the predicted object category
      vs = vs * os.unsqueeze(1)

      # tensor with values from 0 to 199 (inclusive)
      ids = torch.arange(b.shape[0])

      results[-1].update({'verb_scores': vs.to('cpu'), 'sub_ids': ids[:ids.shape[0] // 2], # ids[:100]
                          'obj_ids': ids[ids.shape[0] // 2:]}) # ids[100:]
    return results
    
    

class PostProcess(nn.Module):
  """ This module converts the model's output into the format expected by the coco api"""

  def __init__(self, processor_dct=None):
    super().__init__()
    # For instance segmentation using UQR module
    self.processor_dct = processor_dct

  @torch.no_grad()
  def forward(self, outputs, target_sizes):
    """ Perform the computation

    Parameters:
        outputs: raw outputs of the model
        target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                      For evaluation, this must be the original image size (before any data augmentation)
                      For visualization, this should be the image size after data augment, but before padding
    """

    if self.processor_dct is not None:
      out_logits, out_bbox, out_vector = outputs['pred_logits'], outputs['pred_boxes'], outputs['pred_vectors']
    else:
      out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

    assert len(out_logits) == len(target_sizes)
    assert target_sizes.shape[1] == 2

    prob = out_logits.sigmoid()
    topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), 100, dim=1)
    scores = topk_values
    topk_boxes = topk_indexes // out_logits.shape[2]
    labels = topk_indexes % out_logits.shape[2]
    boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # Added for Instance Segmentation
    if self.processor_dct is not None:
      n_keep = self.processor_dct.n_keep
      vectors = torch.gather(out_vector, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, n_keep))
    ###########

    # and from relative [0, 1] to absolute [0, height] coordinates
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1).to(torch.float32)
    boxes = boxes * scale_fct[:, None, :]

    # Added for Instance Segmentation
    if self.processor_dct is not None:
      masks = []
      n_keep, gt_mask_len = self.processor_dct.n_keep, self.processor_dct.gt_mask_len
      b, r, c = vectors.shape
      for bi in range(b):
        outputs_masks_per_image = []
        for ri in range(r):
          # here visual for training
          idct = np.zeros((gt_mask_len ** 2))
          idct[:n_keep] = vectors[bi, ri].cpu().numpy()
          idct = self.processor_dct.inverse_zigzag(idct, gt_mask_len, gt_mask_len)
          re_mask = cv2.idct(idct)
          max_v = np.max(re_mask)
          min_v = np.min(re_mask)
          re_mask = np.where(re_mask > (max_v + min_v) / 2., 1, 0)
          re_mask = torch.from_numpy(re_mask)[None].float()
          outputs_masks_per_image.append(re_mask)
        outputs_masks_per_image = torch.cat(outputs_masks_per_image, dim=0).to(out_vector.device)
        # here padding local mask to global mask
        outputs_masks_per_image = retry_if_cuda_oom(paste_masks_in_image)(
          outputs_masks_per_image,  # N, 1, M, M
          boxes[bi],
          (img_h[bi], img_w[bi]),
          threshold=0.5,
        )
        outputs_masks_per_image = outputs_masks_per_image.unsqueeze(1).cpu()
        masks.append(outputs_masks_per_image)
    ###########

    if self.processor_dct is None:
      results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    else:
      results = [{'scores': s, 'labels': l, 'boxes': b, 'masks': m} for s, l, b, m in zip(scores, labels, boxes, masks)]

    return results


class PostProcessSegm(nn.Module):
  def __init__(self, threshold=0.5, processor_dct=None):
    super().__init__()
    self.threshold = threshold
    self.processor_dct = processor_dct

  @torch.no_grad()
  def forward(self, results, outputs, orig_target_sizes, max_target_sizes):
    return results