import numpy as np
import torch.nn as nn
import torch.nn.functional as func
import torch

EPS = 1e-8


class Criterion(nn.Module):
    @staticmethod
    def forward(logits, labels, data_ids):
        pass


class BceLoss(Criterion):
    @staticmethod
    def forward(logits, labels, data_ids=None):
        logits = logits.view(-1)  # [B * C]
        labels = labels.view(-1)  # [B * C]

        observed_mask = torch.logical_or(labels == 1, labels == 0)
        observed_logits = logits[observed_mask]
        observed_labels = labels[observed_mask]

        loss = func.binary_cross_entropy_with_logits(observed_logits, observed_labels.float())
        return loss

