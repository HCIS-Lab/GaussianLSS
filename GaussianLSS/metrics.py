import torch
from torchmetrics import Metric

from typing import List
import numpy as np

class BaseIoUMetric(Metric):
    """
    Computes intersection over union at given thresholds
    """
    def __init__(self, thresholds=np.array([0.4, 0.45, 0.5])): 
        super().__init__(dist_sync_on_step=False)
        # super().__init__()
        thresholds = torch.FloatTensor(thresholds)
        self.add_state('thresholds', default=thresholds, dist_reduce_fx='mean')
        self.add_state('tp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fp', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')
        self.add_state('fn', default=torch.zeros_like(thresholds), dist_reduce_fx='sum')

    def update(self, pred, label):
        label = label.detach().bool().reshape(-1)
        pred = pred[:, None] >= self.thresholds[None]
        label = label[:, None]
        self.tp += (pred & label).sum(0)
        self.fp += (pred & ~label).sum(0)
        self.fn += (~pred & label).sum(0)
        #self.tn += (~pred & ~label).sum(0)

    def compute(self):
        ious = self.tp / (self.tp + self.fp + self.fn + 1e-7)
        return ious

    def show_result(self):
        thresholds = self.thresholds.squeeze(0)
        return {f'@{t.item():.2f}': {'tp:':self.tp, 'fp:':self.tp,'fn:':self.fn} for i, t in enumerate(thresholds)}
    
    def compute_recall(self):
        thresholds = self.thresholds.squeeze(0)
        recalls = self.tp / (self.tp + self.fn + 1e-7)
        
        return {f'@{t.item():.2f}': i.item() for t, i in zip(thresholds, recalls)}

class IoUMetric(BaseIoUMetric):
    def __init__(self, min_visibility = 0, key= 'bev'):
        """
        label_indices:
            transforms labels (c, h, w) to (len(labels), h, w)
            see config/experiment/* for examples

        min_visibility:
            passing "None" will ignore the visibility mask
            otherwise uses visibility values to ignore certain labels
            visibility mask is in order of "increasingly visible" {1, 2, 3, 4, 255 (default)}
            see https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/schema_nuscenes.md#visibility
        """
        super().__init__()

        self.min_visibility = min_visibility
        self.key = key

    def update(self, pred, batch):
        pred = pred[self.key].clone().detach().sigmoid()
        target = batch[self.key]

        if self.min_visibility > 0:
            mask = batch[f"{self.key}_visibility"] >= self.min_visibility
            mask = mask[:, None].expand_as(pred)                                            # b c h w
            pred = pred[mask]                                                               # m
            target = target[mask]                                                             # m
        else:
            pred = pred.reshape(-1)
            
        return super().update(pred, target)
    
    def compute(self):
        ious = super().compute()
        max_iou, indice = torch.max(ious, dim=0)
        max_iou = torch.round(max_iou, decimals=4)
        return {f"IoU_{self.key}": max_iou, "Max threshold": self.thresholds[indice]}