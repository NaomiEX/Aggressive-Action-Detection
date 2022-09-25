# altered

- dataset_file (default is hico-det instead of coco)
- set_cost_class (name changed to set_cost_obj_class)
- cls_loss_coef (name changed to obj_cls_loss_coef)

# removed

- coco_path

# added

- train_path (as a replacement to coco_path)
- test_path
- train_anno_path
- test_anno_path
- inter_token_num
- inter_cross_indices
- num_verb_classes
- num_obj_classes
- subject_category_id
- set_cost_verb_class
- verb_cls_loss_coef
- root_path
