import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import Accuracy, AUROC
from metrics.fairness import DemParityMetric, EqualizedOdds, EqualOpportunity
from models.mlp import MLP


class GenericModel(MLP):
    def __init__(self, *xargs, **kwargs):
        super(GenericModel, self).__init__(*xargs, **kwargs)

    def log_metrics(self, preds, y, s, loss, stage):
        super().log_metrics(preds, y, s, loss, stage)
        
