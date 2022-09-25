# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# DN-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]


import torch
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F


def prepare_for_cdn(dn_args, training, num_queries, num_classes, hidden_dim, label_enc):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn, nn.Embedding(101, hidden_dim)
        :return:
        """
    if training:
        # targets are the ground truth targets for each sample
        # dn_number is the number of queries per group, default is 100 
        #   (i.e. 100 for positive noised, i.e. we expect reconstruction and 100 for negative noised, i.e. we expect no-object)
        # label_noise_ratio is refers to noise injected to object category labels, default is 0.5
        # box_noise_scale is the scale of the noise for bbox, default is 0.4
        targets, dn_number, label_noise_ratio, box_noise_scale = dn_args
        # positive and negative denoised queries
        dn_number = dn_number * 2
        
        # generate tensor of 1s with shapes similar to the object labels for each of the target images,
        # for ex. if the objects in a particular image has labels [2, 4] => [1, 1]
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
        batch_size = len(known)
        known_num = [sum(k) for k in known] # sum up the number of objects per image
        if int(max(known_num)) == 0: # if target has no objects, set number of denoised queries to 1
            dn_number = 1
        else:
            # if >= 100 total queries across both positive and negative-noised queries
            if dn_number >= 100:
                # divide the number of queries by double of the largest number of objects in the batch,
                # for ex. if the largest number of objects present in an image in the batch is 4
                # divide dn_number by 8
                dn_number = dn_number // (int(max(known_num) * 2))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0: # set a minimum number of denoised queries to be 1
            dn_number = 1
            
        # concatenates the known tensors by the first dimension
        # for ex. if the first image in the batch has two objects and the second has three:
        # unmask_bbox = unmask_label = [1, 1, 1, 1, 1]
        unmask_bbox = unmask_label = torch.cat(known) # shape: [total_N_o], 
                                                      # where total_N_o is the total number of objects in all images in the batch
        
        # concatenate the labels and boxes of all images in the batch
        labels = torch.cat([t['labels'] for t in targets]) # shape: [total_N_o]
        boxes = torch.cat([t['boxes'] for t in targets]) # shape: [total_N_o, 4]
        
        # fill a tensor of shape identical to [N_o] with the batch index for every image in the batch
        # for ex. if the first image in the batch has two objects and the second has three:
        # batch_idx = [0,0,1,1,1]
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)]) # [total_N_o]

        # since unmask_bbox and unmask_label are 1-tensors, all elements are non-zero
        # thus torch.nonzero just gets the indices of all elements, the .view then flattens it, for ex.
        # [0, 1, 2, 3, 4]
        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        # repeat the set of indices for 2*number of queries and flatten, for ex.
        # [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
        known_indice = known_indice.repeat(2 * dn_number, 1).view(-1) # shape: [total_N_o * (2*dn_number)]
        
        # repeat the gt labels, batch indices, and gt bbox coordinates for every single query and flatten
        known_labels = labels.repeat(2 * dn_number, 1).view(-1) # shape: [total_N_o * (2*dn_number)]
        known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1) # shape: [total_N_o * (2*dn_number)]
        known_bboxs = boxes.repeat(2 * dn_number, 1) # shape: [total_N_o * (2*dn_number)]
        
        ## clone so that the noise does not affect the original gt labels and bboxes
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        ## adding noise to labels
        if label_noise_ratio > 0: # if we also want to denoise labels
            p = torch.rand_like(known_labels_expaned.float()) # randomly generate floats in range [0,1), sample from uniform distribution
            
            # (p < (label_noise_ratio * 0.5)) creates a tensor where the corresponding element of p is 1 if it is less than half the label_noise_ratio
            # by default, this is 0.5, so if p < 1/4, element is a 1, otherwise it is a 0,
            # chosen indice is then the flattened tensor of all indices where p < 1/4
            chosen_indice = torch.nonzero(p < (label_noise_ratio * 0.5)).view(-1) # shape: [N_noise]
            # randomly generate a valid class in range [0,num_classes) to fill a tensor of the same shape as chosen_indice,
            # i.e. randomly generate a valid class category for each p < 1/4 (default)
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # [N_noise]
            # 0 is the dimension, chosen_indice are the indices within the dimension to replace, new_label is the value to replace with
            # for ex. if known_labels_expaned = [4, 9, 18, 0, 2, 4, 9, 18, 0, 2, 4, 9, 18, 0, 2, 4, 9, 18, 0, 2]
            #         and if chosen_indice = [4, 6, 9]
            #         and if new_label = [7, 12, 5]
            #         known_labels_expaned becomes = [4, 9, 18, 0, *7*, 4, *12*, 18, 0, *5*, 4, 9, 18, 0, 2, 4, 9, 18, 0, 2]
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
            
        # max no. of objects across all images in the batch
        single_pad = int(max(known_num))

        pad_size = int(single_pad * 2 * dn_number)
        # (torch.tensor(range(len(boxes)))) -> generates a tensor of range [0, total_N_o), ex. [0, 1, 2, 3, 4]
        # the unsqueeze and repeat, duplicates this for every query, for ex. if dn_number=4
        # [[0, 1, 2, 3, 4],
        #  [0, 1, 2, 3, 4],
        #  [0, 1, 2, 3, 4],
        #  [0, 1, 2, 3, 4]]
        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        # (torch.tensor(range(dn_number))) -> generates a tensor of range [0, dn_number), ex. [0, 1, 2, 3]
        # multiplied by the number of objects * 2, ex. [0, 10, 20, 30]
        # unsqueeze -> [[0], [10], [20], [30]]
        # add with positive_idx, first row gets added with 0, second row added by 10, third row added by 20, etc.
        # for ex.
        # [[0,   1,  2,  3,  4],
        #  [10, 11, 12, 13, 14],
        #  [20, 21, 22, 23, 24],
        #  [30, 31, 32, 33, 34]]
        positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
        positive_idx = positive_idx.flatten() # shape: [dn_number * total_N_o]
        # get the other half missing from positive_idx
        # for ex. 
        # [[5,   6,  7,  8,  9],
        #  [15, 16, 17, 18, 19],
        #  [25, 26, 27, 28, 29],
        #  [35, 36, 37, 38, 39]]
        negative_idx = positive_idx + len(boxes)
        
        # if we are doing denoising training with bboxes
        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs) # shape: [total_N_o, 4]
            
            # convert from [cx, cy, w, h] -> [x1, y1, x2, y2]
            # here lambda_2 = 1/2
            # x1_ = cx-w/2, y1_ = cy-h/2  NOTE: these are the starting x,y coordinates of the object, i.e. x1, y1
            known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
            # x2_ = cx+w/2, y2 = cy+h/2 NOTE: these are the end x,y coords of the bbox, i.e. x2, y2
            known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs) # shape: [total_N_o, 4]
            # x1_diff = w/2, y1_diff = h/2
            diff[:, :2] = known_bboxs[:, 2:] / 2 
            # x2_diff = w/2, y2_diff = h/2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            # generate a tensor randomly filled with either 1 or -1
            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0 # shape: [total_N_o, 4]
            # generate random floats in range [0, 1) sampled from uniform distribution
            rand_part = torch.rand_like(known_bboxs) # shape: [total_N_o, 4]
            # 1.0 is lambda_1
            # noise > lambda_1 is considered a negative sample
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign # randomly flip signs
            # tensor with values [-w/5, w/5] and [-h/5, h/5] (default)
            # add noise
            # added with known_bbox_, x1_ range = [x1- w/5, x1 + w/5], (y1_range similar replace x with y)
            #                         x2_range = [x2-w/5, x2+w/5] (y2_range similar replace x with y)
            known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                                  diff).cuda() * box_noise_scale
            # clamp to [0, 1] because the x and y values are normalized against the width and height of the image
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            
            # convert the noised bboxes from (x1, y1, x2, y2) -> (cx, cy, w, h)
            known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]

        m = known_labels_expaned.long().to('cuda')
        # encode the noisy object categories using an nn.Embedding() module
        input_label_embed = label_enc(m) # shape: [total_N_o * (2*dn_number), hidden_dim]
        
        # change from range (0, 1) -> (-inf, inf)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        ## these are needed because different images in the batch have different number of images, 
        # so we start with a blank canvas full of 0s usign the max no. of objects and later 
        # fill in the parts where there is a corresponding element
        padding_label = torch.zeros(pad_size, hidden_dim).cuda() # shape: [max_N_o * 2 * dn_number, hidden_dim]
        padding_bbox = torch.zeros(pad_size, 4).cuda() # shape: [max_N_o * 2 * dn_number, 4]

        input_query_label = padding_label.repeat(batch_size, 1, 1) # shape: [B, max_N_o * 2 * dn_number, hidden_dim]
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1) # shape: [B, max_N_o * 2 * dn_number, 4]

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num): # if max. no. of objects is not 0
            # for ex. if first image has 2 objects, second has 3 objects:
            # [0, 1, 0, 1, 2]
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num]) 
            # for ex. if dn_number=2, and single_pad=max. no. objects=3
            # [0, 1, 0, 1, 2, 3, 4, 3, 4, 5, 6, 7, 6, 7, 8, 9, 10, 9, 10, 11]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
        if len(known_bid):
            ## fill in the tensor of 0s where there is a corresponding value
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        # total number of queries (noised + non-noised): 
        # max. no. of objects * 2 * dn_number + num queries (these are the non-noised queries)
        tgt_size = pad_size + num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        
        # non-noised queries cannot see noised (gt) queries
        attn_mask[pad_size:, :pad_size] = True
        # mask so that noised groups cannot see each other
        for i in range(dn_number):
            if i == 0:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            if i == dn_number - 1:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
            else:
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

        # variables about the denoising
        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': dn_number,
        }
    else:

        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        dn_meta = None

    return input_query_label, input_query_bbox, attn_mask, dn_meta


def dn_post_process(outputs_class, outputs_coord, dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    if dn_meta and dn_meta['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :dn_meta['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :dn_meta['pad_size'], :]
        outputs_class = outputs_class[:, :, dn_meta['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, dn_meta['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1]}
        if aux_loss:
            out['aux_outputs'] = _set_aux_loss(output_known_class, output_known_coord)
        dn_meta['output_known_lbs_bboxes'] = out
    return outputs_class, outputs_coord


