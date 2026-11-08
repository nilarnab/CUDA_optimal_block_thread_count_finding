"""Losses for soft per-configuration performance targets."""

from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("training.losses requires PyTorch") from exc


class PerformanceTargetLoss(nn.Module):
    def __init__(self, kind: str = "bce") -> None:
        super().__init__()
        if kind not in {"bce", "mse"}:
            raise ValueError("loss kind must be 'bce' or 'mse'")
        self.kind = kind

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.kind == "bce":
            return F.binary_cross_entropy_with_logits(logits, targets)
        return F.mse_loss(torch.sigmoid(logits), targets)

