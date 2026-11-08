"""Variable-length token sequence encoder with optional attention pooling."""

from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
except ImportError as exc:  # pragma: no cover
    raise ImportError("encoders.lstm_encoder requires PyTorch") from exc


class LSTMEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        *,
        layers: int = 1,
        bidirectional: bool = False,
        attention: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        self._output_size = hidden_size * (2 if bidirectional else 1)
        self.attention = nn.Linear(self._output_size, 1) if attention else None

    @property
    def output_size(self) -> int:
        return self._output_size

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            embeddings,
            lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (hidden, _) = self.lstm(packed)
        if self.attention is None:
            if self.lstm.bidirectional:
                return torch.cat((hidden[-2], hidden[-1]), dim=-1)
            return hidden[-1]

        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        steps = torch.arange(output.size(1), device=lengths.device).unsqueeze(0)
        mask = steps < lengths.unsqueeze(1)
        attention_logits = self.attention(output).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(attention_logits, dim=1)
        return torch.sum(output * weights.unsqueeze(-1), dim=1)

