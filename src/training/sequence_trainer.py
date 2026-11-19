"""Use the existing training loop with soft targets or measured runtime regret."""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

from training.losses import PerformanceTargetLoss
from training.supervised_trainer import SupervisedTrainer, TrainerConfig


class ExpectedRuntimeRegretLoss(nn.Module):
    def forward(self, logits: torch.Tensor, runtimes: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape != runtimes.shape:
            raise ValueError("logits and runtimes must have matching [batch, actions] shapes")
        if not bool(torch.isfinite(runtimes).all()) or bool((runtimes <= 0).any()):
            raise ValueError("runtimes must be finite and positive")
        regret = runtimes / runtimes.min(dim=1, keepdim=True).values - 1
        return (logits.softmax(dim=-1) * regret).sum(dim=-1).mean()


class SequenceTrainer(SupervisedTrainer):
    """Adapt the label tensor so both objectives use the unchanged training loop."""

    def __init__(self, model: nn.Module, config: TrainerConfig) -> None:
        if config.loss not in {"bce", "mse", "regret"}:
            raise ValueError("loss must be bce, mse, or regret")
        super().__init__(model, replace(config, loss="bce"))
        self.config = config
        self.loss = (
            ExpectedRuntimeRegretLoss()
            if config.loss == "regret" else PerformanceTargetLoss(config.loss)
        )

    def _move(self, batch: dict) -> dict:
        batch = super()._move(batch)
        if self.config.loss == "regret":
            batch["targets"] = batch["runtimes"]
        return batch

    @torch.no_grad()
    def evaluate(self, batches):
        # Weight by samples, including a smaller final batch, for comparable losses.
        weighted_loss = 0.0
        count = 0
        logits_all, runtimes_all = [], []
        from training.metrics import runtime_metrics

        self.model.eval()
        for batch in batches:
            batch = self._move(batch)
            logits = self.model(batch["tokens"], batch["lengths"], batch["context"])
            size = logits.size(0)
            weighted_loss += float(self.loss(logits, batch["targets"]).item()) * size
            count += size
            logits_all.append(logits.cpu())
            runtimes_all.append(batch["runtimes"].cpu())
        if not count:
            raise ValueError("evaluation loader produced no batches")
        return weighted_loss / count, runtime_metrics(
            torch.cat(logits_all), torch.cat(runtimes_all)
        )
