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

    def final_state_only(self):
        rows = torch.arange(self.states.size(0), device=self.device)
        states = self.states[rows, self.lengths - 1].unsqueeze(1)
        lengths = torch.ones_like(self.lengths)
        return AdaptiveKernelSequence(
            states, lengths, self.gates.new_ones((len(rows), 1)), lengths.float()
        )

    def __getitem__(self, selection):
        lengths = self.lengths[selection]
        width = int(lengths.max())
        return AdaptiveKernelSequence(
            self.states[selection, :width], lengths, self.gates[selection, :width],
            self.expected_lengths[selection],
        )


class AdaptiveConfigurationModel(SequenceConfigurationModel):
    temperature = 2.0 / 3.0
    lower = -0.1
    upper = 1.1

    def __init__(self, embedding_matrix, context_size, action_count, **kwargs):
        super().__init__(embedding_matrix, context_size, action_count, **kwargs)
        hidden = self.kernel_encoder.hidden_size
        self.gate_network = nn.Sequential(
            nn.Linear(2 * hidden, 32), nn.Tanh(), nn.Linear(32, 1)
        )

        nn.init.normal_(self.gate_network[-1].weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gate_network[-1].bias, -2.0)

    def _gate_values(self, log_alpha):
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

    def compress(self, sequence):
        states, lengths = sequence.states, sequence.lengths
        steps = torch.arange(states.size(1), device=states.device)[None, :]
        valid = steps < lengths[:, None]
        final = steps == lengths[:, None] - 1
        last = states[torch.arange(states.size(0), device=states.device), lengths - 1]
        features = torch.cat((states, last[:, None, :].expand_as(states)), dim=-1)
        gates, probability = self._gate_values(self.gate_network(features).squeeze(-1))

        gates = torch.where(final, torch.ones_like(gates), gates).masked_fill(~valid, 0)
        expected = 1 + (probability * (valid & ~final)).sum(dim=1)
        keep = gates > 0
        selected_states = [states[index, mask] for index, mask in enumerate(keep)]
        selected_gates = [gates[index, mask] for index, mask in enumerate(keep)]
        return AdaptiveKernelSequence(
            pad_sequence(selected_states, batch_first=True), keep.sum(dim=1),
            pad_sequence(selected_gates, batch_first=True), expected,
        )

    def encode_kernel(self, tokens, lengths):
        return self.compress(super().encode_kernel(tokens, lengths))

    def score_cached(self, kernel_sequence, context):
        states, lengths = kernel_sequence.states, kernel_sequence.lengths
        context_vector = self.context_encoder(context)
        fused = torch.cat(
            (states, context_vector[:, None, :].expand(-1, states.size(1), -1)), dim=-1
        )
        token_logits = self.scorer(fused)
        valid = torch.arange(states.size(1), device=states.device)[None, :] < lengths[:, None]
        attention = self.attention(fused).squeeze(-1)

        attention = attention + kernel_sequence.gates.clamp_min(torch.finfo(states.dtype).tiny).log()
        weights = attention.masked_fill(~valid, -torch.inf).softmax(dim=1)
        return (token_logits * weights[:, :, None]).sum(dim=1)
