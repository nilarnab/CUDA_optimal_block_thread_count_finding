import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


_TOKEN_PATTERN = re.compile(
    r'!\"(?:\\.|[^\"])*\"|@[-a-zA-Z$._0-9]+|%[-a-zA-Z$._0-9]+|'
    r'[-+]?(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)|'
    r'[A-Za-z_][A-Za-z0-9_.$-]*|==|!=|<=|>=|->|[-+*/%=<>()[\]{},:!]'
)


class LLVMTokenizer:
    def __init__(self, *, normalize_identifiers=True, normalize_numbers=True):
        self.normalize_identifiers = normalize_identifiers
        self.normalize_numbers = normalize_numbers

    def tokenize(self, text):
        text = re.sub(r";.*$", "", text, flags=re.MULTILINE)
        output = []
        for token in _TOKEN_PATTERN.findall(text):
            if self.normalize_identifiers and token.startswith("%"):
                token = "%ID"
            elif self.normalize_identifiers and token.startswith("@"):
                token = "@SYMBOL"
            elif self.normalize_numbers and re.fullmatch(
                r"[-+]?(?:0x[0-9A-Fa-f]+|\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", token
            ):
                token = "<NUM>"
            output.append(token)
        return output

    def tokenize_file(self, path):
        return self.tokenize(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Vocabulary:
    token_to_id: dict[str, int]

    PAD = "<PAD>"
    UNK = "<UNK>"

    @classmethod
    def build(cls, documents, *, min_frequency=1, max_size=None):
        counts = Counter(token for document in documents for token in document)
        candidates = [
            token for token, count in counts.items() if count >= min_frequency
        ]
        candidates.sort(key=lambda token: (-counts[token], token))
        if max_size is not None:
            candidates = candidates[: max(0, max_size - 2)]
        mapping = {cls.PAD: 0, cls.UNK: 1}
        mapping.update({token: index + 2 for index, token in enumerate(candidates)})
        return cls(mapping)

    @property
    def padding_id(self):
        return self.token_to_id[self.PAD]

    def encode(self, tokens):
        unknown = self.token_to_id[self.UNK]
        return tuple(self.token_to_id.get(token, unknown) for token in tokens)

    def save(self, path):
        Path(path).write_text(
            json.dumps(self.token_to_id, indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({str(token): int(index) for token, index in data.items()})
