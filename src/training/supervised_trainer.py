"""A small reusable supervised trainer for any compatible scorer model."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import statistics
import time
from typing import Callable, Iterable

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("training.supervised_trainer requires PyTorch") from exc

from training.losses import PerformanceTargetLoss
from training.metrics import RuntimeMetrics, runtime_metrics


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: float | None = 1.0
    loss: str = "bce"
    device: str = "cpu"
    verbose: bool = False


@dataclass(frozen=True)
class InferenceLatency:
    """Single-sample selector latency with model inputs already prepared."""

    mean_ms: float
    median_ms: float
    p95_ms: float
    samples_per_second: float


class SupervisedTrainer:
    def __init__(self, model: nn.Module, config: TrainerConfig) -> None:
        self.model = model.to(config.device)
        self.config = config
        self.loss = PerformanceTargetLoss(config.loss)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def _move(self, batch: dict) -> dict:
        return {
            key: value.to(self.config.device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }

    def train_epoch(self, batches: Iterable[dict]) -> float:
        self.model.train()
        total_loss = 0.0
        count = 0
        for batch in batches:
            batch = self._move(batch)
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(batch["tokens"], batch["lengths"], batch["context"])
            loss = self.loss(logits, batch["targets"])
            loss.backward()
            if self.config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip
                )
            self.optimizer.step()
            total_loss += float(loss.item())
            count += 1
        if count == 0:
            raise ValueError("training loader produced no batches")
        return total_loss / count

    @torch.no_grad()
    def evaluate(self, batches: Iterable[dict]) -> tuple[float, RuntimeMetrics]:
        self.model.eval()
        losses: list[float] = []
        logits_all: list[torch.Tensor] = []
        runtimes_all: list[torch.Tensor] = []
        for batch in batches:
            batch = self._move(batch)
            logits = self.model(batch["tokens"], batch["lengths"], batch["context"])
            losses.append(float(self.loss(logits, batch["targets"]).item()))
            logits_all.append(logits.cpu())
            runtimes_all.append(batch["runtimes"].cpu())
        if not losses:
            raise ValueError("evaluation loader produced no batches")
        metrics = runtime_metrics(torch.cat(logits_all), torch.cat(runtimes_all))
        return sum(losses) / len(losses), metrics

    def fit(
        self,
        train_batches: Iterable[dict],
        validation_batches: Iterable[dict],
        *,
        epoch_callback: Callable[[dict], None] | None = None,
    ):
        history = []
        for epoch in range(1, self.config.epochs + 1):
            training_loss = self.train_epoch(train_batches)
            validation_loss, metrics = self.evaluate(validation_batches)
            history.append(
                {
                    "epoch": epoch,
                    "training_loss": training_loss,
                    "validation_loss": validation_loss,
                    "performance_score": metrics.performance_score,
                    "mean_regret": metrics.mean_regret,
                    "exact_optimal_rate": metrics.exact_optimal_rate,
                }
            )
            if self.config.verbose:
                values = history[-1]
                print(
                    "[training:epoch] "
                    f"{epoch}/{self.config.epochs} "
                    f"train_loss={values['training_loss']:.6f} "
                    f"validation_loss={values['validation_loss']:.6f} "
                    f"performance={values['performance_score']:.6f} "
                    f"regret={values['mean_regret']:.6f} "
                    f"exact_optimal={values['exact_optimal_rate']:.6f}"
                )
            if epoch_callback is not None:
                epoch_callback(history[-1])
        return history

    @torch.no_grad()
    def benchmark_inference(
        self,
        batches: Iterable[dict],
        *,
        warmup: int = 3,
        repetitions: int = 5,
    ) -> InferenceLatency:
        """Benchmark batch-size-one model forward + argmax latency.

        Tokenization, host-to-device transfer, LLVM compilation, and CUDA
        kernel execution are intentionally excluded.
        """

        if warmup < 0 or repetitions < 1:
            raise ValueError("warmup must be non-negative and repetitions positive")
        self.model.eval()
        samples = []
        for batch in batches:
            batch = self._move(batch)
            for index in range(batch["tokens"].size(0)):
                length = int(batch["lengths"][index].item())
                samples.append(
                    (
                        batch["tokens"][index : index + 1, :length],
                        batch["lengths"][index : index + 1],
                        batch["context"][index : index + 1],
                    )
                )
        if not samples:
            raise ValueError("latency benchmark received no samples")

        def synchronize() -> None:
            device = samples[0][0].device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elif device.type == "mps":
                torch.mps.synchronize()

        first = samples[0]
        for _ in range(warmup):
            self.model(*first).argmax(dim=-1)
        synchronize()

        measurements_ms = []
        for sample in samples:
            for _ in range(repetitions):
                synchronize()
                started = time.perf_counter_ns()
                self.model(*sample).argmax(dim=-1)
                synchronize()
                measurements_ms.append((time.perf_counter_ns() - started) / 1e6)

        ordered = sorted(measurements_ms)
        p95_index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
        mean_ms = statistics.fmean(measurements_ms)
        return InferenceLatency(
            mean_ms=mean_ms,
            median_ms=statistics.median(measurements_ms),
            p95_ms=ordered[p95_index],
            samples_per_second=1000.0 / mean_ms,
        )

    def save_checkpoint(self, path: str | Path, **metadata) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "trainer_config": self.config.__dict__,
                "metadata": metadata,
            },
            path,
        )

