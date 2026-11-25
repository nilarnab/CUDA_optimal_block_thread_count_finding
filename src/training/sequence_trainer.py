from dataclasses import replace

import torch
from torch import nn

from training.losses import PerformanceTargetLoss
from training.supervised_trainer import SupervisedTrainer


class ExpectedRuntimeRegretLoss(nn.Module):
    def forward(self, logits, runtimes):
        regret = runtimes / runtimes.min(dim=1, keepdim=True).values - 1
        return (logits.softmax(dim=-1) * regret).sum(dim=-1).mean()


class SequenceTrainer(SupervisedTrainer):
    def __init__(self, model, config):
        super().__init__(model, replace(config, loss="bce"))
        self.config = config
        self.loss = (
            ExpectedRuntimeRegretLoss()
            if config.loss == "regret" else PerformanceTargetLoss(config.loss)
        )

    def _move(self, batch):
        batch = super()._move(batch)
        if self.config.loss == "regret":
            batch["targets"] = batch["runtimes"]
        return batch

    @torch.no_grad()
    def evaluate(self, batches):

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
        return weighted_loss / count, runtime_metrics(
            torch.cat(logits_all), torch.cat(runtimes_all)
        )
