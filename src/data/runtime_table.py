import csv
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from launch.configuration_catalog import Configuration


@dataclass(frozen=True, order=True)
class RuntimeKey:
    kernel_id: str
    problem_size: tuple[int, ...]


class RuntimeTable:
    def __init__(self):
        self._measurements = defaultdict(list)

    def add(self, key, configuration, runtime):
        runtime = float(runtime)
        self._measurements[(key, configuration)].append(runtime)

    def runtime(self, key, configuration):
        values = self._measurements.get((key, configuration))
        if not values:
            raise KeyError(f"no runtime for {key!r} and {configuration!r}")
        return float(statistics.median(values))

    def runtime_vector(self, key, catalog):
        return tuple(self.runtime(key, config) for config in catalog)

    def has_complete_context(self, key, catalog):
        return all((key, config) in self._measurements for config in catalog)

    def contexts(self):
        return tuple(sorted({key for key, _ in self._measurements}))

    @classmethod
    def from_csv(
        cls,
        path,
        *,
        kernel_column="kernel_id",
        problem_columns=("problem_size",),
        block_columns=("block_x", "block_y", "block_z"),
        runtime_column="runtime_us",
        defaults=None,
    ):
        table = cls()
        defaults = {"block_y": 1, "block_z": 1, **(defaults or {})}
        with Path(path).open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)

            for row in reader:
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
        return table

    def write_medians(self, path):
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
