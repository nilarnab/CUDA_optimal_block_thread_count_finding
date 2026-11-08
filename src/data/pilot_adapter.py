"""Convert a correctness-checked LS-CAT pilot run into model inputs."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


NLP_THREAD_COUNTS = tuple(range(64, 1025, 64))


@dataclass(frozen=True)
class PilotPreparationSummary:
    kernels: int
    problem_sizes: int
    contexts: int
    actions_per_context: int
    runtime_rows: int
    skipped_non_1d_rows: int


def _abstract_thread_count(block_x: int, block_y: int) -> int | None:
    """Map a native 1D launch orientation to its abstract thread count.

    Square and other 2D shapes intentionally return ``None``. Orientation is
    a property of the launch adapter, while the NLP baseline predicts one of
    the sixteen scalar thread-count levels.
    """

    if block_y == 1 and block_x in NLP_THREAD_COUNTS:
        return block_x
    if block_x == 1 and block_y in NLP_THREAD_COUNTS:
        return block_y
    return None


def prepare_pilot_dataset(
    run_directory: str | Path,
    output_directory: str | Path,
) -> PilotPreparationSummary:
    run_directory = Path(run_directory)
    output_directory = Path(output_directory)
    action_path = run_directory / "action_table.csv"
    manifest_path = run_directory / "source" / "kernels" / "manifest.json"
    if not action_path.is_file():
        raise FileNotFoundError(action_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)

    required = {"kernel_id", "size", "block_x", "block_y", "median_ms"}
    normalized_rows: list[dict[str, object]] = []
    physical_mappings: dict[str, dict[str, list[int]]] = defaultdict(dict)
    coverage: dict[tuple[str, int], set[int]] = defaultdict(set)
    skipped = 0
    with action_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"action table is missing columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                kernel_id = row["kernel_id"]
                problem_size = int(row["size"])
                block_x = int(row["block_x"])
                block_y = int(row["block_y"])
                median_ms = float(row["median_ms"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid action-table row {row_number}: {exc}") from exc
            if not math.isfinite(median_ms) or median_ms <= 0:
                raise ValueError(
                    f"invalid action-table row {row_number}: median_ms must be positive"
                )
            thread_count = _abstract_thread_count(block_x, block_y)
            if thread_count is None:
                skipped += 1
                continue
            context = (kernel_id, problem_size)
            if thread_count in coverage[context]:
                raise ValueError(
                    f"duplicate abstract action {thread_count} for {context!r}"
                )
            coverage[context].add(thread_count)
            native_shape = [block_x, block_y, 1]
            previous_shape = physical_mappings[kernel_id].get(str(thread_count))
            if previous_shape is not None and previous_shape != native_shape:
                raise ValueError(
                    f"inconsistent native shape for {kernel_id!r}, action {thread_count}: "
                    f"{previous_shape!r} versus {native_shape!r}"
                )
            physical_mappings[kernel_id][str(thread_count)] = native_shape
            normalized_rows.append(
                {
                    "kernel_id": kernel_id,
                    "problem_size": problem_size,
                    "block_x": thread_count,
                    "block_y": 1,
                    "block_z": 1,
                    "runtime_us": median_ms * 1000.0,
                }
            )

    expected = set(NLP_THREAD_COUNTS)
    incomplete = {
        context: sorted(expected.difference(actions))
        for context, actions in coverage.items()
        if actions != expected
    }
    if incomplete:
        preview = list(incomplete.items())[:5]
        raise ValueError(f"incomplete 16-action contexts: {preview!r}")
    if not normalized_rows:
        raise ValueError("action table produced no one-dimensional runtime rows")

    kernel_ids = sorted({row["kernel_id"] for row in normalized_rows})
    problem_sizes = sorted({int(row["problem_size"]) for row in normalized_rows})
    expected_context_count = len(kernel_ids) * len(problem_sizes)
    if len(coverage) != expected_context_count:
        raise ValueError(
            f"expected {expected_context_count} kernel/size contexts, found {len(coverage)}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_entries = {entry["id"]: entry for entry in manifest["kernels"]}
    missing_sources = sorted(set(kernel_ids).difference(manifest_entries))
    if missing_sources:
        raise ValueError(f"manifest is missing kernels: {missing_sources}")

    output_directory.mkdir(parents=True, exist_ok=True)
    cuda_directory = output_directory / "cuda"
    cuda_directory.mkdir(parents=True, exist_ok=True)
    # Every pilot kernel includes this filename. The one required compatibility
    # alias is harmless for the remaining kernels.
    (cuda_directory / "includes.h").write_text(
        "#pragma once\nusing uint = unsigned int;\n", encoding="utf-8"
    )
    for kernel_id in kernel_ids:
        entry = manifest_entries[kernel_id]
        source_path = run_directory / "source" / "kernels" / entry["path"]
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        (cuda_directory / f"{kernel_id}.cu").write_text(
            source_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    normalized_rows.sort(
        key=lambda row: (
            str(row["kernel_id"]),
            int(row["problem_size"]),
            int(row["block_x"]),
        )
    )
    runtime_path = output_directory / "runtimes.csv"
    fields = (
        "kernel_id",
        "problem_size",
        "block_x",
        "block_y",
        "block_z",
        "runtime_us",
    )
    with runtime_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(normalized_rows)

    (output_directory / "launch_mappings.json").write_text(
        json.dumps(physical_mappings, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = PilotPreparationSummary(
        kernels=len(kernel_ids),
        problem_sizes=len(problem_sizes),
        contexts=len(coverage),
        actions_per_context=len(NLP_THREAD_COUNTS),
        runtime_rows=len(normalized_rows),
        skipped_non_1d_rows=skipped,
    )
    (output_directory / "pilot_preparation.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
