from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from launch.configuration_catalog import ConfigurationCatalog

from .runtime_table import RuntimeTable


@dataclass(frozen=True)
class PreparedSample:
    kernel_id: str
    token_ids: tuple[int, ...]
    problem_size: tuple[float, ...]
    runtimes: tuple[float, ...]
    target_quality: tuple[float, ...]
    optimal_action: int


class KernelRuntimeDataset(Sequence[PreparedSample]):
    def __init__(
        self,
        runtime_table: RuntimeTable,
        catalog: ConfigurationCatalog,
        token_ids_by_kernel: Mapping[str, Sequence[int]],
        *,
        kernel_ids: Iterable[str] | None = None,
        target_exponent: float = 12.0,
    ) -> None:
        
        allowed = None if kernel_ids is None else set(kernel_ids)
        samples: list[PreparedSample] = []
        for key in runtime_table.contexts():
            if allowed is not None and key.kernel_id not in allowed:
                continue
            if key.kernel_id not in token_ids_by_kernel:
                continue
            if not runtime_table.has_complete_context(key, catalog):
                continue
            token_ids = tuple(int(value) for value in token_ids_by_kernel[key.kernel_id])
            runtimes = runtime_table.runtime_vector(key, catalog)
            best = min(runtimes)
            samples.append(
                PreparedSample(
                    kernel_id=key.kernel_id,
                    token_ids=token_ids,
                    problem_size=tuple(float(value) for value in key.problem_size),
                    runtimes=runtimes,
                    target_quality=tuple(
                        (best / runtime) ** target_exponent for runtime in runtimes
                    ),
                    optimal_action=min(range(len(runtimes)), key=runtimes.__getitem__),
                )
            )
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> PreparedSample:
        return self._samples[index]


def collate_samples(samples, *, padding_id):

    import torch

    lengths = torch.tensor([len(sample.token_ids) for sample in samples], dtype=torch.long)
    max_length = int(lengths.max())
    tokens = torch.full((len(samples), max_length), padding_id, dtype=torch.long)
    for row, sample in enumerate(samples):
        tokens[row, : len(sample.token_ids)] = torch.tensor(sample.token_ids)
    return {
        "kernel_ids": [sample.kernel_id for sample in samples],
        "tokens": tokens,
        "lengths": lengths,
        "context": torch.tensor([sample.problem_size for sample in samples], dtype=torch.float32),
        "runtimes": torch.tensor([sample.runtimes for sample in samples], dtype=torch.float32),
        "targets": torch.tensor(
            [sample.target_quality for sample in samples], dtype=torch.float32
        ),
        "optimal_actions": torch.tensor(
            [sample.optimal_action for sample in samples], dtype=torch.long
        ),
    }

