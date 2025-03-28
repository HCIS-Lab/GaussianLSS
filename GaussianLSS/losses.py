import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
from fvcore.nn import sigmoid_focal_loss

logger = logging.getLogger(__name__)

class SpatialRegressionLoss(torch.nn.Module):
    def __init__(self, norm, min_visibility, ignore_index):
        super(SpatialRegressionLoss, self).__init__()
        self.min_visibility = min_visibility
        self.ignore_index = ignore_index
        # center:2, offset: 1
        self.norm = norm
        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, pred, target, visibility, eps=1e-6):
        mask = torch.ones_like(target, dtype=torch.bool)
        if self.min_visibility > 0:
            vis_mask = visibility >= self.min_visibility
            vis_mask = vis_mask[:, None]
            mask = mask * vis_mask

        if self.ignore_index is not None:
            mask = mask * (target != self.ignore_index)
        
        loss = self.loss_fn(pred, target, reduction='none')
        return (loss * mask).sum() / (mask.sum() + eps)

class CenterLoss(SpatialRegressionLoss):
    def __init__(self, min_visibility=0, ignore_index=None, key=''):
        super().__init__(2, min_visibility, ignore_index)
        self.key = key

    def forward(self, prediction, batch):
        key = f"{self.key}_center"
        prediction = prediction[key].sigmoid()
        target = batch[key]

        assert len(prediction.shape) == 4, 'Must be a 4D tensor'
        visibility = batch[f"{self.key}_visibility"]
        return super().forward(prediction, target, visibility)

class OffsetLoss(SpatialRegressionLoss):
    def __init__(self, min_visibility=0, ignore_index=None, key=''):
        super().__init__(1, min_visibility, ignore_index)
        self.key = key

    def forward(self, prediction, batch):
        key = f"{self.key}_offset"
        prediction = prediction[key]
        target = batch[key]

        assert len(prediction.shape) == 4, 'Must be a 4D tensor'
        visibility = batch[f"{self.key}_visibility"]
        return super().forward(prediction, target, visibility)
    
class SigmoidFocalLoss(torch.nn.Module):
    def __init__(
        self,
        alpha=-1.0,
        gamma=2.0,
        reduction='mean'
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, label):
        return sigmoid_focal_loss(pred, label, self.alpha, self.gamma, self.reduction)

class BinarySegmentationLoss(SigmoidFocalLoss):
    def __init__(
        self,
        min_visibility=0,
        alpha=-1.0,
        gamma=2.0,
        key='bev',
    ):
        super().__init__(alpha=alpha, gamma=gamma, reduction='none')
        
        self.min_visibility = min_visibility
        self.key = key

    def forward(self, pred_dict, batch, eps=1e-6):
        pred = pred_dict[self.key]
        target = batch[self.key]
        loss = super().forward(pred, target)

        mask = torch.ones_like(target, dtype=torch.bool)
        if self.min_visibility > 0:
            visibility = batch[f"{self.key}_visibility"]
            vis_mask = visibility >= self.min_visibility
            vis_mask = vis_mask[:, None]
            mask = mask * vis_mask
        
        return (loss * mask).sum() / (mask.sum() + eps)

class MultipleLoss(torch.nn.ModuleDict):
    """
    losses = MultipleLoss({'bce': torch.nn.BCEWithLogitsLoss(), 'bce_weight': 1.0})
    loss, unweighted_outputs = losses(pred, label)
    """
    def __init__(self, modules_or_weights):
        modules = dict()
        weights = dict()
        learnable_weights = dict()

        # Parse only the weights
        for key, v in modules_or_weights.items():
            if isinstance(v, float):
                k = key.replace('_weight', '')
                if v == -1:
                    weights[k] =  0.5 if k not in ['visible', 'ped'] else 10.0
                    learnable_weights[k] = nn.Parameter(torch.tensor(0.0), requires_grad=True)
                else:
                    weights[k] = v

        # Parse the loss functions
        for key, v in modules_or_weights.items():
            if not isinstance(v, float):
                modules[key] = v

        super().__init__(modules)

        self._weights = weights
        self.learnable_weights = torch.nn.ParameterDict(learnable_weights)

    def forward(self, pred, batch):
        outputs = dict()
        weights = dict()

        for k, v in self.items():

            if k =='learnable_weights':
                continue
            elif k != 'Set':
                outputs[k] = v(pred, batch)
            else:
                if 'pred_logits' not in pred:
                    continue
                out = v(pred, batch)
                for k2, v2 in out.items():
                    outputs[k2] = v2
        # outputs = {k: v(pred, batch) for k, v in self.items()}
        loss = []
        for k, o in outputs.items():
            loss_weight = self._weights[k]
            if k in self.learnable_weights:
                loss_weight = (1 / torch.exp(self.learnable_weights[k])) * loss_weight
                weights[k] = loss_weight
                uncertainty = self.learnable_weights[k] * 0.5
            else:
                uncertainty = 0.0
            single_loss = loss_weight * o + uncertainty
            outputs[k] = single_loss
            loss.append(single_loss)

        return sum(loss), outputs, weights