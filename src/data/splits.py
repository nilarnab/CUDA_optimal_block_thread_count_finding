"""Deterministic, leakage-resistant dataset splitting by kernel identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class KernelSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class TrainValidationSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]


def _score(kernel_id: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{kernel_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def split_kernel_ids(
    kernel_ids: Iterable[str],
    *,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    seed: int = 0,
) -> KernelSplit:
    """Split unique kernel IDs so one kernel cannot occur in multiple splits."""

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train and validation fractions must sum to less than 1")

    ordered = sorted(set(kernel_ids), key=lambda value: (_score(value, seed), value))
    count = len(ordered)
    train_end = round(count * train_fraction)
    validation_end = train_end + round(count * validation_fraction)
    return KernelSplit(
        train=tuple(ordered[:train_end]),
        validation=tuple(ordered[train_end:validation_end]),
        test=tuple(ordered[validation_end:]),
    )


def split_train_validation_kernel_ids(
    kernel_ids: Iterable[str],
    *,
    validation_kernels: int = 2,
    seed: int = 0,
) -> TrainValidationSplit:
    """Create two disjoint partitions while keeping every kernel intact."""

    ordered = sorted(set(kernel_ids), key=lambda value: (_score(value, seed), value))
    if validation_kernels < 1:
        raise ValueError("validation_kernels must be positive")
    if validation_kernels >= len(ordered):
        raise ValueError("validation must leave at least one training kernel")
    return TrainValidationSplit(
        train=tuple(ordered[:-validation_kernels]),
        validation=tuple(ordered[-validation_kernels:]),
    )

