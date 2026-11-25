import torch
from torch import nn


class FrozenFastTextEncoder(nn.Module):
    def __init__(self, embedding_matrix, *, padding_id=0):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.float(), freeze=True, padding_idx=padding_id
        )

    @property
    def output_size(self):
        return self.embedding.embedding_dim

    def forward(self, token_ids):
        return self.embedding(token_ids)


def embedding_matrix_from_fasttext(model_path, token_to_id, *, padding_token="<PAD>"):
    import fasttext
    model = fasttext.load_model(str(model_path))
    dimensions = model.get_dimension()
    matrix = torch.zeros((len(token_to_id), dimensions), dtype=torch.float32)
    for token, index in token_to_id.items():
        if token != padding_token:
            matrix[index] = torch.tensor(model.get_word_vector(token))
    return matrix
