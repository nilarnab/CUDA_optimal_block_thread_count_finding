"""Encode problem dimensions or other structured launch context."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("encoders.context_encoder requires PyTorch") from exc


class ContextEncoder(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self._output_size = output_size
        self.network = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.LayerNorm(output_size),
        )

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        # Log scaling keeps large matrix dimensions numerically manageable.
        return self.network(torch.log1p(context.clamp_min(0)))

