import torch
from torch import nn
from torch.nn import functional as F


class PerformanceTargetLoss(nn.Module):
    def __init__(self, kind="bce"):
        super().__init__()
        self.kind = kind

    def forward(self, logits, targets):
        if self.kind == "bce":
            return F.binary_cross_entropy_with_logits(logits, targets)
        return F.mse_loss(torch.sigmoid(logits), targets)
