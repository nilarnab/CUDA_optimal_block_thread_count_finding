import torch
from torch import nn


class ConcatenationFusion(nn.Module):
    def forward(self, *representations):
        return torch.cat(representations, dim=-1)
