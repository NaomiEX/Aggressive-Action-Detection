#!/bin/bash
python main.py \
--method vidt \
--epochs 50 \
--backbone_name swin_nano \
--pre_trained ./params/swin_nano_patch4_window7_224.pth \
--batch_size 4 \
--aux_loss True \
--with_obj_box_refine True \
--with_sub_box_refine True \
--det_token_num 100 \
--inter_token_num 100 \
--dataset_file 'surveillance' \
--num_verb_classes 4 \
--num_obj_classes 4 \
--root_path './data/surveillance' \
--output_dir 'surveillance/run' \
--use_nms \
--predict_interaction_vector