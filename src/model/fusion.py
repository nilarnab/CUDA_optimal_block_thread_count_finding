"""Composable representation-fusion strategies."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("model.fusion requires PyTorch") from exc


class ConcatenationFusion(nn.Module):
    def forward(self, *representations: torch.Tensor) -> torch.Tensor:
        if not representations:
            raise ValueError("at least one representation is required")
        return torch.cat(representations, dim=-1)

