"""Measured runtimes indexed by kernel, problem, and launch configuration."""

from __future__ import annotations

import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from launch.configuration_catalog import Configuration, ConfigurationCatalog


@dataclass(frozen=True, order=True)
class RuntimeKey:
    """A workload context independent of the selected configuration."""

    kernel_id: str
    problem_size: tuple[int, ...]


class RuntimeTable:
    """Stores repeated measurements and exposes their median runtime.

    The CSV reader is intentionally schema-configurable so a future dataset does
    not need to use LS-CAT's column names.
    """

    def __init__(self) -> None:
        self._measurements: dict[
            tuple[RuntimeKey, Configuration], list[float]
        ] = defaultdict(list)

    def add(
        self,
        key: RuntimeKey,
        configuration: Configuration,
        runtime: float,
    ) -> None:
        runtime = float(runtime)
        if not math.isfinite(runtime) or runtime <= 0:
            raise ValueError(f"runtime must be finite and positive, got {runtime!r}")
        self._measurements[(key, configuration)].append(runtime)

    def runtime(self, key: RuntimeKey, configuration: Configuration) -> float:
        values = self._measurements.get((key, configuration))
        if not values:
            raise KeyError(f"no runtime for {key!r} and {configuration!r}")
        return float(statistics.median(values))

    def runtime_vector(
        self,
        key: RuntimeKey,
        catalog: ConfigurationCatalog,
    ) -> tuple[float, ...]:
        return tuple(self.runtime(key, config) for config in catalog)

    def has_complete_context(
        self,
        key: RuntimeKey,
        catalog: ConfigurationCatalog,
    ) -> bool:
        return all((key, config) in self._measurements for config in catalog)

    def contexts(self) -> tuple[RuntimeKey, ...]:
        return tuple(sorted({key for key, _ in self._measurements}))

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        kernel_column: str = "kernel_id",
        problem_columns: Sequence[str] = ("problem_size",),
        block_columns: Sequence[str] = ("block_x", "block_y", "block_z"),
        runtime_column: str = "runtime_us",
        defaults: Mapping[str, int] | None = None,
    ) -> "RuntimeTable":
        table = cls()
        defaults = {"block_y": 1, "block_z": 1, **(defaults or {})}
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("runtime CSV has no header")
            required = {kernel_column, runtime_column, *problem_columns}
            missing = required.difference(reader.fieldnames)
            if missing:
                raise ValueError(f"runtime CSV is missing columns: {sorted(missing)}")

            for row_number, row in enumerate(reader, start=2):
                try:
                    problem_size = tuple(int(row[name]) for name in problem_columns)
                    block = []
                    for name in block_columns:
                        raw = row.get(name, "")
                        block.append(int(raw) if raw not in (None, "") else defaults[name])
                    while len(block) < 3:
                        block.append(1)
                    table.add(
                        RuntimeKey(row[kernel_column], problem_size),
                        Configuration(*block[:3]),
                        float(row[runtime_column]),
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid runtime CSV row {row_number}: {exc}") from exc
        return table

    def write_medians(self, path: str | Path) -> None:
        """Write a normalized table useful for inspecting or caching data."""

        dimensions = max((len(key.problem_size) for key in self.contexts()), default=1)
        problem_headers = (
            ["problem_size"]
            if dimensions == 1
            else [f"problem_size_{index}" for index in range(dimensions)]
        )
        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "kernel_id",
                    *problem_headers,
                    "block_x",
                    "block_y",
                    "block_z",
                    "runtime_us",
                ]
            )
            for (key, config), values in sorted(self._measurements.items()):
                writer.writerow(
                    [
                        key.kernel_id,
                        *key.problem_size,
                        config.x,
                        config.y,
                        config.z,
                        statistics.median(values),
                    ]
                )

