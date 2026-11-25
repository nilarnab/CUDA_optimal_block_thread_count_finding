from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from encoders.context_encoder import ContextEncoder
from encoders.fasttext_encoder import FrozenFastTextEncoder
from model.config_scorer import ConfigurationScorer


@dataclass(frozen=True)
class KernelSequence:
    states: torch.Tensor
    lengths: torch.Tensor

    @property
    def device(self):
        return self.states.device

    def __getitem__(self, selection):
        lengths = self.lengths[selection]
        return KernelSequence(self.states[selection, : int(lengths.max())], lengths)


class SequenceConfigurationModel(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        context_size,
        action_count,
        *,
        hidden_size=128,
        context_features=32,
        hidden_sizes=(256, 128, 64),
        dropout=0.1,
    ):
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

    def encode_kernel(self, tokens, lengths):
        packed = pack_padded_sequence(
            self.token_encoder(tokens), lengths.detach().cpu(),
            batch_first=True, enforce_sorted=False,
        )
        output, _ = self.kernel_encoder(packed)
        states, _ = pad_packed_sequence(output, batch_first=True)
        return KernelSequence(states, lengths.to(states.device))

    def score_cached(self, kernel_sequence, context):
        states, lengths = kernel_sequence.states, kernel_sequence.lengths
        context_vector = self.context_encoder(context)
        fused = torch.cat(
            (states, context_vector.unsqueeze(1).expand(-1, states.size(1), -1)), dim=-1
        )

        token_logits = self.scorer(fused)
        valid = torch.arange(states.size(1), device=states.device)[None, :] < lengths[:, None]
        weights = self.attention(fused).squeeze(-1).masked_fill(~valid, -torch.inf)
        weights = weights.softmax(dim=1)
        return (token_logits * weights.unsqueeze(-1)).sum(dim=1)

    def forward(self, tokens, lengths, context):
        return self.score_cached(self.encode_kernel(tokens, lengths), context)


class FinalStateConfigurationModel(SequenceConfigurationModel):
    def encode_kernel(self, tokens, lengths):
        sequence = super().encode_kernel(tokens, lengths)
        rows = torch.arange(sequence.states.size(0), device=sequence.states.device)
        final_states = sequence.states[rows, sequence.lengths - 1].unsqueeze(1)
        return KernelSequence(final_states, torch.ones_like(sequence.lengths))
