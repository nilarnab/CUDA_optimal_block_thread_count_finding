"""Single inference entry point assembling generic selector modules."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
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


@dataclass(frozen=True)
class CachedKernel:
    """Kernel representation computed once and reused across launches."""

    representation: torch.Tensor


class InferenceEngine:
    def __init__(
        self,
        model,
        tokenizer: LLVMTokenizer,
        vocabulary: Vocabulary,
        catalog: ConfigurationCatalog,
        *,
        device: str = "cpu",
        problem_columns: Sequence[str] = (),
    ) -> None:
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.catalog = catalog
        self.device = device
        self.problem_columns = tuple(problem_columns)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, vocabulary_path: str | Path, *, device: str = "cpu"
    ) -> InferenceEngine:
        """Load the selector and the exact vocabulary used to train its embeddings."""
        from model.config_scorer import build_model

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = saved["metadata"]
        architecture = metadata.get("architecture", "baseline")
        if metadata.get("format_version") not in (None, 1) or (
            architecture == "adaptive" and metadata.get("adaptive_experiment_version") != 1
        ):
            raise ValueError("unsupported selector checkpoint format")
        context_size = metadata["context_size"]
        columns = tuple(metadata.get("problem_columns", ("problem_size",)))
        if len(columns) != context_size:
            raise ValueError("checkpoint problem columns do not match its context size")
        catalog = ConfigurationCatalog(Configuration(**shape) for shape in metadata["action_catalog"])
        state = saved["model_state"]
        matrix = state["token_encoder.embedding.weight"]
        vocabulary = Vocabulary.load(vocabulary_path)
        ids = vocabulary.token_to_id
        if (ids.get(Vocabulary.PAD) != 0 or ids.get(Vocabulary.UNK) != 1
                or sorted(ids.values()) != list(range(matrix.size(0)))):
            raise ValueError("vocabulary IDs do not match the checkpoint embedding table")
        model = build_model(matrix, context_size, len(catalog), architecture,
                            metadata.get("lstm_hidden_size", 128))
        model.load_state_dict(state)
        return cls(model, LLVMTokenizer(), vocabulary, catalog, device=device, problem_columns=columns)

    @torch.no_grad()
    def prepare_kernel(self, llvm_ir: str) -> CachedKernel:
        """Run tokenization and sequence encoding once for a known kernel."""

        token_ids = self.vocabulary.encode(self.tokenizer.tokenize(llvm_ir))
        if not token_ids:
            token_ids = (self.vocabulary.token_to_id[Vocabulary.UNK],)
        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=self.device)
        return CachedKernel(self.model.encode_kernel(tokens, lengths))

    @torch.no_grad()
    def predict_cached(
        self, cached_kernel: CachedKernel, problem_size: Sequence[int], *,
        allowed: Sequence[Configuration] | None = None,
    ) -> Prediction:
        """Choose a configuration using only the launch-time predictor path."""

        if self.problem_columns and len(problem_size) != len(self.problem_columns):
            raise ValueError(f"expected problem dimensions in this order: {self.problem_columns}")
        if allowed is None:
            candidates = range(len(self.catalog))
        else:
            permitted = set(allowed)
            candidates = [index for index, config in enumerate(self.catalog) if config in permitted]
            if not candidates:
                raise ValueError("no permitted configuration is present in the checkpoint catalog")
        context = torch.tensor([problem_size], dtype=torch.float32, device=self.device)
        logits = self.model.score_cached(cached_kernel.representation, context)[0]
        scores = tuple(float(value) for value in logits.cpu())
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("model produced a non-finite launch score")
        action_id = max(candidates, key=scores.__getitem__)
        return Prediction(
            action_id=action_id,
            configuration=self.catalog.from_id(action_id),
            scores=scores,
        )

    @torch.no_grad()
    def predict(self, llvm_ir: str, problem_size: Sequence[int]) -> Prediction:
        """Convenience method for uncached one-off predictions."""

        return self.predict_cached(self.prepare_kernel(llvm_ir), problem_size)

