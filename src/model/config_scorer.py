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

    def encode_kernel(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute the kernel representation that can be cached offline."""

        return self.kernel_encoder(self.token_encoder(tokens), lengths)

    def score_cached(
        self, kernel_vector: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Score configurations from a cached kernel vector at launch time."""

        context_vector = self.context_encoder(context)
        return self.scorer(self.fusion(kernel_vector, context_vector))

    def forward(
        self,
        tokens: torch.Tensor,
        lengths: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        return self.score_cached(self.encode_kernel(tokens, lengths), context)


class ContextOnlyConfigurationModel(nn.Module):
    """Ablation that predicts from structured problem dimensions alone."""

    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        context_size: int,
        action_count: int,
        *,
        context_features: int = 32,
        hidden_sizes: Sequence[int] = (256, 128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Keep the frozen matrix in checkpoints so the existing data provenance
        # and evaluation path remain identical across ablations.
        self.token_encoder = FrozenFastTextEncoder(embedding_matrix)
        self.context_encoder = ContextEncoder(context_size, context_features)
        self.scorer = ConfigurationScorer(
            context_features, action_count, hidden_sizes=hidden_sizes, dropout=dropout
        )

    def encode_kernel(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Return an empty cache: kernel identity and tokens are intentionally ignored."""

        if tokens.ndim != 2 or lengths.shape != (tokens.size(0),):
            raise ValueError("expected tokens [batch, steps] and lengths [batch]")
        return torch.empty((tokens.size(0), 0), device=tokens.device)

    def score_cached(
        self, kernel_cache: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        if kernel_cache.ndim != 2 or kernel_cache.size(0) != context.size(0):
            raise ValueError("kernel cache and context batch sizes must match")
        return self.scorer(self.context_encoder(context))

    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return self.score_cached(self.encode_kernel(tokens, lengths), context)

