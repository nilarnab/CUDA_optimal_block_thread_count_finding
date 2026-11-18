"""Score every LSTM timestep before reducing the variable-length sequence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from encoders.context_encoder import ContextEncoder
from encoders.fasttext_encoder import FrozenFastTextEncoder
from model.config_scorer import ConfigurationScorer


@dataclass(frozen=True)
class KernelSequence:
    """A padded cache with lengths, supporting the existing latency benchmark."""

    states: torch.Tensor
    lengths: torch.Tensor

    @property
    def device(self) -> torch.device:
        return self.states.device

    def __getitem__(self, selection: slice) -> KernelSequence:
        lengths = self.lengths[selection]
        return KernelSequence(self.states[selection, : int(lengths.max())], lengths)


class SequenceConfigurationModel(nn.Module):
    def __init__(
        self,
        embedding_matrix: torch.Tensor,
        context_size: int,
        action_count: int,
        *,
        hidden_size: int = 128,
        context_features: int = 32,
        hidden_sizes: tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.token_encoder = FrozenFastTextEncoder(embedding_matrix)
        self.kernel_encoder = nn.LSTM(
            self.token_encoder.output_size, hidden_size, batch_first=True
        )
        self.context_encoder = ContextEncoder(context_size, context_features)
        features = hidden_size + context_features
        self.scorer = ConfigurationScorer(
            features, action_count, hidden_sizes=hidden_sizes, dropout=dropout
        )
        self.attention = nn.Sequential(
            nn.Linear(features, context_features), nn.Tanh(), nn.Linear(context_features, 1)
        )

    def encode_kernel(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> KernelSequence:
        """Cache all hidden states and lengths; no context-dependent pooling yet."""
        if tokens.ndim != 2 or lengths.shape != (tokens.size(0),):
            raise ValueError("expected tokens [batch, steps] and lengths [batch]")
        if lengths.dtype != torch.long or bool(
            ((lengths < 1) | (lengths > tokens.size(1))).any()
        ):
            raise ValueError("lengths must be positive int64 values within the token width")
        packed = pack_padded_sequence(
            self.token_encoder(tokens), lengths.detach().cpu(),
            batch_first=True, enforce_sorted=False,
        )
        output, _ = self.kernel_encoder(packed)
        states, _ = pad_packed_sequence(output, batch_first=True)
        return KernelSequence(states, lengths.to(states.device))

    def score_cached(
        self,
        kernel_sequence: KernelSequence,
        context: torch.Tensor,
    ) -> torch.Tensor:
        states, lengths = kernel_sequence.states, kernel_sequence.lengths
        context_vector = self.context_encoder(context)
        fused = torch.cat(
            (states, context_vector.unsqueeze(1).expand(-1, states.size(1), -1)), dim=-1
        )
        # Linear layers share weights across timesteps; only the last width is fixed.
        token_logits = self.scorer(fused)
        valid = torch.arange(states.size(1), device=states.device)[None, :] < lengths[:, None]
        weights = self.attention(fused).squeeze(-1).masked_fill(~valid, -torch.inf)
        weights = weights.softmax(dim=1)
        return (token_logits * weights.unsqueeze(-1)).sum(dim=1)

    def forward(
        self, tokens: torch.Tensor, lengths: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        return self.score_cached(self.encode_kernel(tokens, lengths), context)


class FinalStateConfigurationModel(SequenceConfigurationModel):
    """Matched sequence scorer that always retains only the final LSTM state."""

    def encode_kernel(
        self, tokens: torch.Tensor, lengths: torch.Tensor
    ) -> KernelSequence:
        sequence = super().encode_kernel(tokens, lengths)
        rows = torch.arange(sequence.states.size(0), device=sequence.states.device)
        final_states = sequence.states[rows, sequence.lengths - 1].unsqueeze(1)
        return KernelSequence(final_states, torch.ones_like(sequence.lengths))
