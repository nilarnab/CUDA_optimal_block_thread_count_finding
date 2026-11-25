import torch
from torch import nn


class ContextEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self._output_size = output_size
        self.network = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.LayerNorm(output_size),
        )

    @property
    def output_size(self):
        return self._output_size

    def forward(self, context):

        return self.network(torch.log1p(context.clamp_min(0)))
