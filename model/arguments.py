# ------------------------------------------------------------------------
# DETR
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
# Additionally modified by NAVER Corp. for ViDT
# ------------------------------------------------------------------------

import argparse

def str2bool(v, bool):

    if isinstance(v, bool):
        return v
    if v.lower() in ('true', 't'):
        return True
    elif v.lower() in ('false', 'f'):
        return False
    else:
        argparse.ArgumentTypeError('Boolean value expected.')

def get_args_parser():
    parser = argparse.ArgumentParser('Set ViHOI-DET', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--eval_size', default=800, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--eval_extra', action='store_true')
    
    parser.add_argument('--optimizer', default='AdamW', type=str)
    
    # weight standardization
    parser.add_argument('--weight_standardization', action='store_true')

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='warmupcos', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step", options:"step", "warmupcos"')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                         help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                         help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                         help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-7, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # * model setting
    parser.add_argument('--backbone_name', default='swin_tiny', type=str,
                        help="Name of the deit backbone to use")
    parser.add_argument('--pre_trained', default='imagenet', type=str,
                        help="set imagenet pretrained model path if not train yolos from scatch")

    # * Matcher
    parser.add_argument('--set_cost_obj_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    parser.add_argument('--set_cost_verb_class', default=2, type=float, help="Verb class coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--obj_cls_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=2.5, type=float)
    parser.add_argument('--giou_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    parser.add_argument('--verb_vec_loss_coef', default=1, type=float)
    # * HOI-DET loss coefficients
    parser.add_argument('--verb_cls_loss_coef', default=1, type=float)

    # * Dataset
    # parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--dataset_file', default='hico-det')
    # parser.add_argument('--coco_path', default='/home/Research/MyData/COCO2017', type=str)
    parser.add_argument('--root_path', default='data/surveillance', type=str)
    parser.add_argument('--train_path', default='data/surveillance/train_2021', type=str)
    parser.add_argument('--test_path', default='data/surveillance/test', type=str)
    parser.add_argument('--train_anno_path', default='data/surveillance/train_2021.json', type=str)
    parser.add_argument('--test_anno_path', default='data/surveillance/test_anno.json', type=str)
    # parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    # * Device and Log
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=False, type=lambda x: (str(x).lower() == 'true'), help='eval mode')
    parser.add_argument('--resume', default='', help='resume from checkpoint')

    # * Training setup
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:3457', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--distributed', action='store_true', help='')
    parser.add_argument('--num_workers', default=2, type=int)

    # * Pos encodig
    parser.add_argument('--position_embedding', default='sine', type=str)

    # * Transformer
    parser.add_argument('--pos_dim', default=256, type=int, help="Size of the embeeding for pos")
    parser.add_argument('--reduced_dim', default=256, type=int, help="Size of the embeddings for head")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int, # Deform-DETR: 1024, DETR: 2048
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--without_iaa', action='store_true', help="don't use instance-aware attention in neck decoder")
    parser.add_argument('--single_branch_decoder', action='store_true', 
                        help="collapse instance and interaction branch to a single decoder branch")
    parser.add_argument('--predict_interaction_vector', action='store_true', 
                        help='predict the interaction vector between the subject and object')

    # * Deformable Attention
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--focal_alpha', default=0.25, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')


    ####### ViDT Params
    parser.add_argument('--method', default='vidt', type=str, help='method names in {vidt, vidt_wo_neck}')
    parser.add_argument("--det_token_num", default=100, type=int, help="Number of det token in the body backbone")
    parser.add_argument('--cross_indices', default=[3], nargs='*', type=int, help='stage ids for [DET x PATCH] cross-attention')
    # * HOI detection params
    parser.add_argument('--inter_token_num', default=100, type=int, help="Number of [INTER] tokens in the body backbone")
    parser.add_argument('--inter_cross_indices', default=[3], nargs='*', type=int, 
                        help='stage ids for [INTER x DET] cross-attention')
    parser.add_argument('--inter_patch_cross_indices', default=[], nargs='*', type=int, 
                        help='stage ids for [INTER x PATCH] cross-attention')

    # * Auxiliary Techniques
    parser.add_argument('--aux_loss', default=False, type=lambda x: (str(x).lower() == 'true'), help='auxiliary decoding loss')
    # parser.add_argument('--with_box_refine', default=False, type=lambda x: (str(x).lower() == 'true'), help='iterative box refinement')
    parser.add_argument('--with_obj_box_refine', default=False, type=lambda x: (str(x).lower() == 'true'), 
                        help='iterative box refinement for objects')
    parser.add_argument('--with_sub_box_refine', default=False, type=lambda x: (str(x).lower() == 'true'), 
                        help='iterative box refinement for subjects')
    

    # * Distillation with token matching
    parser.add_argument('--distil_loss_coef', default=4.0, type=float, help="Distillation coefficient")
    parser.add_argument('--distil_model', default=None, type=str, help="Distillation model in {vidt_tiny, vidt_small, vidt-base}")
    parser.add_argument('--distil_model_path', default=None, type=str, help="Distillation model path to load")
    #######
    
    ####### HOI Params
    parser.add_argument('--num_verb_classes', type=int, default=117)
    parser.add_argument('--num_obj_classes', type=int, default=80)
    parser.add_argument('--subject_category_id', default=0, type=int, help="id for the subject in the HOI, note that if ids are 1-indexed, provide the subject id on a 0-indexed scale")
    
    #######
    
    # * NMS
    parser.add_argument('--use_nms', action='store_true')
    parser.add_argument('--nms_thresh', default=0.5, type=float)

    ####### For ViDT+

    ## EPFF
    parser.add_argument('--epff', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='use of EPFF module for pyramid feature fusion')

    ## UQR
    parser.add_argument('--masks', default=False, action='store_true', help="Train segmentation head if the flag is provided")
    parser.add_argument('--with_vector', default=False, type=bool)
    parser.add_argument('--n_keep', default=256, type=int,
                        help="Number of coeffs to be remained")
    parser.add_argument('--gt_mask_len', default=128, type=int,
                        help="Size of target mask")
    parser.add_argument('--vector_loss_coef', default=3.0, type=float)
    parser.add_argument('--vector_hidden_dim', default=256, type=int,
                        help="Size of the vector embeddings (dimension of the transformer)")
    parser.add_argument('--no_vector_loss_norm', default=False, action='store_true')
    parser.add_argument('--activation', default='relu', type=str, help="Activation function to use")
    parser.add_argument('--checkpoint', default=False, action='store_true')
    parser.add_argument('--vector_start_stage', default=0, type=int)
    parser.add_argument('--num_machines', default=1, type=int)
    parser.add_argument('--loss_type', default='l1', type=str)
    parser.add_argument('--dcn', default=False, action='store_true')

    ## New losses
    # iou-aware
    parser.add_argument('--iou_aware', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='use of iou-aware loss')
    parser.add_argument('--iouaware_loss_coef', default=2, type=float)
    # token label
    parser.add_argument('--token_label', default=False, type=lambda x: (str(x).lower() == 'true'),
                        help='use of token label loss')
    parser.add_argument('--token_loss_coef', default=2, type=float)
    #######

    # * Logs
    parser.add_argument('--n_iter_to_acc', default=1, type=int, help='gradient accumulation step size')
    parser.add_argument('--print_freq', default=10, type=int, help='number of iteration to print training logs')

    return parser
