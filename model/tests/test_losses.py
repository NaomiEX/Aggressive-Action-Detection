import pytest
import torch

from methods.vidt.criterion import SetCriterion
from methods.vidt.matcher import HungarianMatcher

# defaults
SET_COST_OBJ_CLS = 2
SET_COST_BBOX = 5
SET_COST_GIOU = 2
SET_COST_VERB_CLASS = 2
NUM_CLASSES = 3
OBJ_CLS_LOSS_COEF = 1
VERB_CLS_LOSS_COEF = 1
BBOX_LOSS_COEF = 2.5
GIOU_LOSS_COEF = 1
NUM_INTERACTIONS=2

@pytest.fixture
def weight_dict():
    return {
        'loss_obj_ce': OBJ_CLS_LOSS_COEF,
        'loss_verb_ce': VERB_CLS_LOSS_COEF,
        'loss_sub_bbox': BBOX_LOSS_COEF,
        'loss_obj_bbox': BBOX_LOSS_COEF,
        'loss_sub_giou': GIOU_LOSS_COEF,
        'loss_obj_giou': GIOU_LOSS_COEF
    }

@pytest.fixture
def losses_in_use():
    return ['labels', 'boxes', 'verbs']

@pytest.fixture
def targets():
    return [{
        'obj_boxes': torch.Tensor([[0.2, 0.8, 0.6, 0.5],
                                [0.8, 0.1, 0.1, 0.2]]),
        'sub_boxes': torch.Tensor([[0.5, 0.1, 0.3, 0.7],
                                [0.5, 0.1, 0.3, 0.7]]),
        'obj_labels': torch.Tensor([0, 2, 1]).long(),
        'verb_labels': torch.Tensor([[0,1,0,1], [1, 0, 0, 0]])

    }]

@pytest.fixture
def target_indices():
    return [(torch.Tensor([0,1]).long(), torch.Tensor([0,1]).long())]

@pytest.fixture
def output_good():
    return {
        'pred_logits': torch.Tensor([[[0.9, 0.01, 0.05],
                                    [0.01, 0.15, 0.99],
                                    [0.06, 1.12, 0.01]]]),
        'pred_boxes': torch.Tensor([[[0.201, 0.78, 0.09, 0.45],
                                    [0.82, 0.1, 0.07, 0.21]]]),
        'pred_verb_logits': torch.Tensor([[[0.001, 0.999, 0.008, 1.03],
                                        [1.41, 0.2, 0.01, 0.0004]]]),
        'pred_sub_boxes': torch.Tensor([[[0.48, 0.09, 0.31, 0.72],
                                        [0.49, 0.1, 0.28, 0.69]]])
    }
    
    
@pytest.fixture
def output_bad():
    return {
        'pred_logits': torch.Tensor([[[0.1, 0.81, 0.65],
                                    [0.91, 0.05, 0.001],
                                    [0.66, 0.02, 0.71]]]),
        'pred_boxes': torch.Tensor([[[0.001, 0.28, 0.09, 0.05],
                                    [0.22, 0.84, 0.7, 0.41]]]),
        'pred_verb_logits': torch.Tensor([[[0.501, 0.009, 0.208, 0.03],
                                        [0.41, 4.2, 0.81, 0.7004]]]),
        'pred_sub_boxes': torch.Tensor([[[0.28, 0.99, 0.11, 0.22],
                                        [0.89, 0.14, 0.78, 0.49]]])
    }

@pytest.fixture
def criterion(weight_dict, losses_in_use):
    matcher = HungarianMatcher(cost_class=SET_COST_OBJ_CLS,
                            cost_bbox=SET_COST_BBOX,
                            cost_giou=SET_COST_GIOU,
                            cost_verb_class=SET_COST_VERB_CLASS)
    criterion = SetCriterion(NUM_CLASSES,matcher, weight_dict, losses_in_use)
    return criterion
    

def test_loss_labels(criterion, output_good, targets, target_indices):
    losses = criterion.loss_labels(output_good, targets, target_indices, NUM_INTERACTIONS)
    assert 'loss_obj_ce' in losses.keys() 
    
def test_loss_labels_good_bad(criterion, output_good, output_bad, targets, target_indices):
    losses = criterion.loss_labels(output_good, targets, target_indices, NUM_INTERACTIONS)
    losses_bad = criterion.loss_labels(output_bad, targets, target_indices, NUM_INTERACTIONS)
    assert losses['loss_obj_ce'] < losses_bad['loss_obj_ce']
    
def test_loss_verb_labels(criterion, output_good, targets, target_indices):
    losses = criterion.loss_verb_labels(output_good, targets, target_indices, NUM_INTERACTIONS)
    assert 'loss_verb_ce' in losses.keys()
    
def test_loss_verb_labels_good_bad(criterion, output_good, output_bad, targets, target_indices):
    losses = criterion.loss_verb_labels(output_good, targets, target_indices, NUM_INTERACTIONS)
    losses_bad = criterion.loss_verb_labels(output_bad, targets, target_indices, NUM_INTERACTIONS)
    assert losses['loss_verb_ce'] < losses_bad['loss_verb_ce']
    
    
def test_loss_boxes(criterion, output_good, targets, target_indices):
    losses = criterion.loss_boxes(output_good, targets, target_indices, NUM_INTERACTIONS)
    print(losses)
    assert all([k in ['loss_sub_bbox', 'loss_obj_bbox', 'loss_sub_giou', 'loss_obj_giou'] for k in losses.keys()])
    
def test_loss_boxes_good_bad(criterion, output_good, output_bad, targets, target_indices):
    losses = criterion.loss_boxes(output_good, targets, target_indices, NUM_INTERACTIONS)
    losses_bad = criterion.loss_boxes(output_bad, targets, target_indices, NUM_INTERACTIONS)
    print(losses)
    print(losses_bad)
    assert all([losses[k] < losses_bad[k] for k in losses.keys()])
    
    