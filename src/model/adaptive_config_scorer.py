"""Input-dependent hard-concrete gates on complete LSTM kernel representations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from model.sequence_config_scorer import KernelSequence, SequenceConfigurationModel


@dataclass(frozen=True)
class AdaptiveKernelSequence(KernelSequence):
    gates: torch.Tensor
    expected_lengths: torch.Tensor

    def __getitem__(self, selection: slice) -> AdaptiveKernelSequence:
        lengths = self.lengths[selection]
        width = int(lengths.max())
        return AdaptiveKernelSequence(
            self.states[selection, :width], lengths, self.gates[selection, :width],
            self.expected_lengths[selection],
        )


class AdaptiveConfigurationModel(SequenceConfigurationModel):
    """Read every token, retain a learned subset, and reuse the sequence scorer."""

    temperature = 2.0 / 3.0
    lower = -0.1
    upper = 1.1

    def __init__(self, embedding_matrix, context_size, action_count, **kwargs):
        super().__init__(embedding_matrix, context_size, action_count, **kwargs)
        hidden = self.kernel_encoder.hidden_size
        self.gate_network = nn.Sequential(
            nn.Linear(2 * hidden, 32), nn.Tanh(), nn.Linear(32, 1)
        )
        nn.init.zeros_(self.gate_network[-1].weight)
        nn.init.constant_(self.gate_network[-1].bias, -2.0)

    def _gate_values(self, log_alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Hard-concrete sampling and expected active count, Louizos et al. (2018)."""
        probability = torch.sigmoid(
            log_alpha - self.temperature * math.log(-self.lower / self.upper)
        )
        if self.training:
            epsilon = torch.finfo(log_alpha.dtype).eps
            uniform = torch.rand_like(log_alpha).clamp(epsilon, 1 - epsilon)
            concrete = torch.sigmoid(
                (uniform.log() - torch.log1p(-uniform) + log_alpha) / self.temperature
            )
        else:
            concrete = log_alpha.sigmoid()
        gates = (concrete * (self.upper - self.lower) + self.lower).clamp(0, 1)
        return gates, probability

    def compress(self, sequence: KernelSequence) -> AdaptiveKernelSequence:
        states, lengths = sequence.states, sequence.lengths
        steps = torch.arange(states.size(1), device=states.device)[None, :]
        valid = steps < lengths[:, None]
        final = steps == lengths[:, None] - 1
        last = states[torch.arange(states.size(0), device=states.device), lengths - 1]
        features = torch.cat((states, last[:, None, :].expand_as(states)), dim=-1)
        gates, probability = self._gate_values(self.gate_network(features).squeeze(-1))
        # Keep the final state so every representation includes a full-input summary.
        gates = torch.where(final, torch.ones_like(gates), gates).masked_fill(~valid, 0)
        expected = 1 + (probability * (valid & ~final)).sum(dim=1)
        keep = gates > 0
        selected_states = [states[index, mask] for index, mask in enumerate(keep)]
        selected_gates = [gates[index, mask] for index, mask in enumerate(keep)]
        return AdaptiveKernelSequence(
            pad_sequence(selected_states, batch_first=True), keep.sum(dim=1),
            pad_sequence(selected_gates, batch_first=True), expected,
        )

    def encode_kernel(self, tokens: torch.Tensor, lengths: torch.Tensor) -> AdaptiveKernelSequence:
        return self.compress(super().encode_kernel(tokens, lengths))

    def score_cached(self, kernel_sequence: AdaptiveKernelSequence, context: torch.Tensor) -> torch.Tensor:
        states, lengths = kernel_sequence.states, kernel_sequence.lengths
        context_vector = self.context_encoder(context)
        fused = torch.cat(
            (states, context_vector[:, None, :].expand(-1, states.size(1), -1)), dim=-1
        )
        token_logits = self.scorer(fused)
        valid = torch.arange(states.size(1), device=states.device)[None, :] < lengths[:, None]
        attention = self.attention(fused).squeeze(-1)
        # Retained continuous gates also weight attention; zero rows were removed upstream.
        attention = attention + kernel_sequence.gates.clamp_min(torch.finfo(states.dtype).tiny).log()
        weights = attention.masked_fill(~valid, -torch.inf).softmax(dim=1)
        return (token_logits * weights[:, :, None]).sum(dim=1)
