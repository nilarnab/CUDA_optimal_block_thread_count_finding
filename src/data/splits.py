from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class KernelSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class TrainValidationSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]


def _score(kernel_id, seed):
    digest = hashlib.sha256(f"{seed}:{kernel_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def split_kernel_ids(kernel_ids, *, train_fraction=0.8, validation_fraction=0.1, seed=0):
    ordered = sorted(set(kernel_ids), key=lambda value: (_score(value, seed), value))
    count = len(ordered)
    train_end = round(count * train_fraction)
    validation_end = train_end + round(count * validation_fraction)
    return KernelSplit(
        train=tuple(ordered[:train_end]),
        validation=tuple(ordered[train_end:validation_end]),
        test=tuple(ordered[validation_end:]),
    )


def split_train_validation_kernel_ids(kernel_ids, *, validation_kernels=2, seed=0):
    ordered = sorted(set(kernel_ids), key=lambda value: (_score(value, seed), value))
    return TrainValidationSplit(
        train=tuple(ordered[:-validation_kernels]),
        validation=tuple(ordered[-validation_kernels:]),
    )


def split_grouped_kernel_ids(
    kernel_ids,
    groups,
    *,
    validation_groups=2,
    training_groups=None,
    seed=0,
):
    kernels = set(kernel_ids)
    if training_groups is None:
        grouped = split_train_validation_kernel_ids(
            groups.values(), validation_kernels=validation_groups, seed=seed
        )
    else:
        unique_groups = sorted(set(groups.values()), key=lambda value: (_score(value, seed), value))
        selected = training_groups + validation_groups
        grouped = TrainValidationSplit(
            train=tuple(unique_groups[:training_groups]),
            validation=tuple(unique_groups[training_groups:selected]),
        )
    return TrainValidationSplit(
        train=tuple(sorted(kernel for kernel in kernels if groups[kernel] in grouped.train)),
        validation=tuple(sorted(kernel for kernel in kernels if groups[kernel] in grouped.validation)),
    )
