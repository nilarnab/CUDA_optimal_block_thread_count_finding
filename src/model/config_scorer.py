"""Shared model that scores every candidate launch configuration."""

from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("model.config_scorer requires PyTorch") from exc

from encoders.context_encoder import ContextEncoder
from encoders.fasttext_encoder import FrozenFastTextEncoder
from encoders.lstm_encoder import LSTMEncoder
from model.fusion import ConcatenationFusion


class ConfigurationScorer(nn.Module):
    def __init__(
        self,
        input_size: int,
        action_count: int,
        *,
        hidden_sizes: Sequence[int] = (256, 128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        sizes = [input_size, *hidden_sizes, action_count]
        layers: list[nn.Module] = []
        for index, (source, target) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(source, target))
            if index < len(sizes) - 2:
                layers.extend((nn.ReLU(), nn.Dropout(dropout)))
        self.network = nn.Sequential(*layers)

    def forward(self, representation: torch.Tensor) -> torch.Tensor:
        return self.network(representation)


class ConfigurationModel(nn.Module):
    """Assembles interchangeable modules into one trainable PyTorch model."""

    def __init__(
        self,
        token_encoder: FrozenFastTextEncoder,
        kernel_encoder: LSTMEncoder,
        context_encoder: ContextEncoder,
        scorer: ConfigurationScorer,
        fusion: ConcatenationFusion | None = None,
    ) -> None:
        super().__init__()
        self.token_encoder = token_encoder
        self.kernel_encoder = kernel_encoder
        self.context_encoder = context_encoder
        self.fusion = fusion or ConcatenationFusion()
        self.scorer = scorer

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        token_vectors = self.token_encoder(tokens)
        kernel_vector = self.kernel_encoder(token_vectors, lengths)
        context_vector = self.context_encoder(context)
        return self.scorer(self.fusion(kernel_vector, context_vector))

