from dataclasses import dataclass
import math

import torch

from frontend.tokenizer import LLVMTokenizer, Vocabulary
from launch.configuration_catalog import Configuration, ConfigurationCatalog


@dataclass(frozen=True)
class Prediction:
    action_id: int
    configuration: Configuration
    scores: tuple[float, ...]


@dataclass(frozen=True)
class CachedKernel:
    representation: torch.Tensor


class InferenceEngine:
    def __init__(
        self,
        model,
        tokenizer,
        vocabulary,
        catalog,
        *,
        device="cpu",
        problem_columns=(),
    ):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.vocabulary = vocabulary
        self.catalog = catalog
        self.device = device
        self.problem_columns = tuple(problem_columns)

    @classmethod
    def from_checkpoint(cls, checkpoint, vocabulary_path, *, device="cpu"):
        from model.config_scorer import build_model

        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        metadata = saved["metadata"]
        architecture = metadata.get("architecture", "baseline")
        context_size = metadata["context_size"]
        columns = tuple(metadata.get("problem_columns", ("problem_size",)))
        catalog = ConfigurationCatalog(Configuration(**shape) for shape in metadata["action_catalog"])
        state = saved["model_state"]
        matrix = state["token_encoder.embedding.weight"]
        vocabulary = Vocabulary.load(vocabulary_path)
        model = build_model(matrix, context_size, len(catalog), architecture,
                            metadata.get("lstm_hidden_size", 128))
        model.load_state_dict(state)
        return cls(model, LLVMTokenizer(), vocabulary, catalog, device=device, problem_columns=columns)

    @torch.no_grad()
    def prepare_kernel(self, llvm_ir):
        token_ids = self.vocabulary.encode(self.tokenizer.tokenize(llvm_ir))
        if not token_ids:
            token_ids = (self.vocabulary.token_to_id[Vocabulary.UNK],)
        tokens = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=self.device)
        return CachedKernel(self.model.encode_kernel(tokens, lengths))

    @torch.no_grad()
    def predict_cached(self, cached_kernel, problem_size, *, allowed=None):
        if allowed is None:
            candidates = range(len(self.catalog))
        else:
            permitted = set(allowed)
            candidates = [index for index, config in enumerate(self.catalog) if config in permitted]
        context = torch.tensor([problem_size], dtype=torch.float32, device=self.device)
        logits = self.model.score_cached(cached_kernel.representation, context)[0]
        scores = tuple(float(value) for value in logits.cpu())
        action_id = max(candidates, key=scores.__getitem__)
        return Prediction(
            action_id=action_id,
            configuration=self.catalog.from_id(action_id),
            scores=scores,
        )

    @torch.no_grad()
    def predict(self, llvm_ir, problem_size):
        return self.predict_cached(self.prepare_kernel(llvm_ir), problem_size)
