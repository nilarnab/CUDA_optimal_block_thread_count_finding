"""Runtime-aware evaluation metrics for configuration selection."""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("training.metrics requires PyTorch") from exc


@dataclass(frozen=True)
class RuntimeMetrics:
    performance_score: float
    mean_regret: float
    exact_optimal_rate: float


def runtime_metrics(logits: torch.Tensor, runtimes: torch.Tensor) -> RuntimeMetrics:
    """Evaluate selected runtime relative to the fastest available runtime."""

    selected_actions = logits.argmax(dim=-1)
    selected = runtimes.gather(1, selected_actions.unsqueeze(1)).squeeze(1)
    optimal, optimal_actions = runtimes.min(dim=1)
    performance = optimal / selected
    regret = (selected - optimal) / optimal
    return RuntimeMetrics(
        performance_score=float(performance.mean().item()),
        mean_regret=float(regret.mean().item()),
        exact_optimal_rate=float((selected_actions == optimal_actions).float().mean().item()),
    )

