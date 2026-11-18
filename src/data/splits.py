"""Deterministic, leakage-resistant dataset splitting by kernel identity."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Mapping


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


def split_grouped_kernel_ids(
    kernel_ids: Iterable[str],
    groups: Mapping[str, str],
    *,
    validation_groups: int = 2,
    training_groups: int | None = None,
    seed: int = 0,
) -> TrainValidationSplit:
    """Reserve complete source groups rather than individual related kernels."""
    kernels = set(kernel_ids)
    if not isinstance(groups, Mapping) or set(groups) != kernels or any(
        not isinstance(g, str) or not g.strip() for g in groups.values()
    ):
        raise ValueError("split groups must map every kernel to a nonempty group name")
    if training_groups is None:
        grouped = split_train_validation_kernel_ids(
            groups.values(), validation_kernels=validation_groups, seed=seed
        )
    else:
        unique_groups = sorted(set(groups.values()), key=lambda value: (_score(value, seed), value))
        if training_groups < 1:
            raise ValueError("training_groups must be positive")
        if validation_groups < 1:
            raise ValueError("validation_groups must be positive")
        selected = training_groups + validation_groups
        if selected > len(unique_groups):
            raise ValueError("requested more training/validation groups than are available")
        grouped = TrainValidationSplit(
            train=tuple(unique_groups[:training_groups]),
            validation=tuple(unique_groups[training_groups:selected]),
        )
    return TrainValidationSplit(
        train=tuple(sorted(kernel for kernel in kernels if groups[kernel] in grouped.train)),
        validation=tuple(sorted(kernel for kernel in kernels if groups[kernel] in grouped.validation)),
    )

