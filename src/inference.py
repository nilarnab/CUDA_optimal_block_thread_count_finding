"""Single inference entry point assembling generic selector modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError("inference requires PyTorch") from exc

from frontend.tokenizer import LLVMTokenizer, Vocabulary
from launch.configuration_catalog import Configuration, ConfigurationCatalog


@dataclass(frozen=True)
class Prediction:
    action_id: int
    configuration: Configuration
    scores: tuple[float, ...]


class InferenceEngine:
    def __init__(
        self,
        model,
        tokenizer: LLVMTokenizer,
        vocabulary: Vocabulary,
        catalog: ConfigurationCatalog,
        *,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.catalog = catalog
        self.device = device

    @torch.no_grad()
    def predict(self, llvm_ir: str, problem_size: Sequence[int]) -> Prediction:
        token_ids = self.vocabulary.encode(self.tokenizer.tokenize(llvm_ir))
        if not token_ids:
            token_ids = (self.vocabulary.token_to_id[Vocabulary.UNK],)
        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=self.device)
        context = torch.tensor([problem_size], dtype=torch.float32, device=self.device)
        logits = self.model(tokens, lengths, context)[0]
        action_id = int(logits.argmax().item())
        return Prediction(
            action_id=action_id,
            configuration=self.catalog.from_id(action_id),
            scores=tuple(float(value) for value in logits.cpu()),
        )

