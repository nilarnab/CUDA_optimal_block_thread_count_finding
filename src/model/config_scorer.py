import torch
from torch import nn

from encoders.context_encoder import ContextEncoder
from encoders.fasttext_encoder import FrozenFastTextEncoder
from encoders.lstm_encoder import LSTMEncoder
from model.fusion import ConcatenationFusion


class ConfigurationScorer(nn.Module):
    def __init__(self, input_size, action_count, *, hidden_sizes=(256, 128, 64), dropout=0.1):
        super().__init__()
        sizes = [input_size, *hidden_sizes, action_count]
        layers = []
        for index, (source, target) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(source, target))
            if index < len(sizes) - 2:
                layers.extend((nn.ReLU(), nn.Dropout(dropout)))
        self.network = nn.Sequential(*layers)

    def forward(self, representation):
        return self.network(representation)


class ConfigurationModel(nn.Module):
    def __init__(self, token_encoder, kernel_encoder, context_encoder, scorer, fusion=None):
        super().__init__()
        self.token_encoder = token_encoder
        self.kernel_encoder = kernel_encoder
        self.context_encoder = context_encoder
        self.fusion = fusion or ConcatenationFusion()
        self.scorer = scorer

    def encode_kernel(self, tokens, lengths):
        return self.kernel_encoder(self.token_encoder(tokens), lengths)

    def score_cached(self, kernel_vector, context):
        context_vector = self.context_encoder(context)
        return self.scorer(self.fusion(kernel_vector, context_vector))

    def forward(self, tokens, lengths, context):
        return self.score_cached(self.encode_kernel(tokens, lengths), context)


class ContextOnlyConfigurationModel(nn.Module):
    def __init__(
        self,
        embedding_matrix,
        context_size,
        action_count,
        *,
        context_features=32,
        hidden_sizes=(256, 128, 64),
        dropout=0.1,
    ):
        super().__init__()

        self.token_encoder = FrozenFastTextEncoder(embedding_matrix)
        self.context_encoder = ContextEncoder(context_size, context_features)
        self.scorer = ConfigurationScorer(
            context_features, action_count, hidden_sizes=hidden_sizes, dropout=dropout
        )

    def encode_kernel(self, tokens, lengths):
        return torch.empty((tokens.size(0), 0), device=tokens.device)

    def score_cached(self, kernel_cache, context):
        return self.scorer(self.context_encoder(context))

    def forward(self, tokens, lengths, context):
        return self.score_cached(self.encode_kernel(tokens, lengths), context)


def build_model(matrix, context_size, action_count, architecture, lstm_hidden_size=128):
    if architecture == "context":
        return ContextOnlyConfigurationModel(matrix, context_size, action_count)
    if architecture == "adaptive":
        from model.adaptive_config_scorer import AdaptiveConfigurationModel

        return AdaptiveConfigurationModel(
            matrix, context_size, action_count, hidden_size=lstm_hidden_size
        )
    if architecture in {"sequence", "final"}:
        from model.sequence_config_scorer import (
            FinalStateConfigurationModel,
            SequenceConfigurationModel,
        )

        model_class = (
            SequenceConfigurationModel
            if architecture == "sequence"
            else FinalStateConfigurationModel
        )
        return model_class(matrix, context_size, action_count, hidden_size=lstm_hidden_size)
    tokens = FrozenFastTextEncoder(matrix)
    kernel = LSTMEncoder(tokens.output_size, lstm_hidden_size, attention=True)
    context = ContextEncoder(context_size, 32)
    return ConfigurationModel(
        tokens, kernel, context,
        ConfigurationScorer(kernel.output_size + context.output_size, action_count),
    )
