"""Frozen FastText token vectors exposed as a PyTorch module."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - clear failure when imported without ML deps
    raise ImportError("encoders.fasttext_encoder requires PyTorch") from exc


class FrozenFastTextEncoder(nn.Module):
    def __init__(self, embedding_matrix: torch.Tensor, *, padding_id: int = 0) -> None:
        super().__init__()
        if embedding_matrix.ndim != 2:
            raise ValueError("embedding_matrix must have shape [vocabulary, dimensions]")
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.float(), freeze=True, padding_idx=padding_id
        )

    @property
    def output_size(self) -> int:
        return self.embedding.embedding_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


def embedding_matrix_from_fasttext(
    model_path: str | Path,
    token_to_id: Mapping[str, int],
    *,
    padding_token: str = "<PAD>",
) -> torch.Tensor:
    """Load native FastText vectors, including subword vectors for unseen tokens."""

    try:
        import fasttext
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "loading a FastText model requires the 'fasttext' optional dependency"
        ) from exc
    model = fasttext.load_model(str(model_path))
    dimensions = model.get_dimension()
    matrix = torch.zeros((len(token_to_id), dimensions), dtype=torch.float32)
    for token, index in token_to_id.items():
        if token != padding_token:
            matrix[index] = torch.tensor(model.get_word_vector(token))
    return matrix

