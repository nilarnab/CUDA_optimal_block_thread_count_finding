"""Prepare the twelve-kernel 1D collection without treating K aliases as kernels."""

from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re

from data.pilot_adapter import NLP_THREAD_COUNTS


def _context_key(row: dict, kernel_field: str) -> tuple:
    kernel = row[kernel_field]
    if any(isinstance(row[name], (bool, float)) for name in ("m", "n", "k", "input_seed_index")):
        raise ValueError("dimensions and seed index must be integers")
    dimensions = tuple(int(row[name]) for name in ("m", "n", "k"))
    seed_index = int(row["input_seed_index"])
    if not isinstance(kernel, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", kernel):
        raise ValueError("invalid base kernel ID")
    if min(dimensions) < 1 or seed_index < 0:
        raise ValueError("M/N/K must be positive and seed index nonnegative")
    return kernel, *dimensions, seed_index


def prepare_one_d_dataset(run_directory: str | Path, output_directory: str | Path) -> dict:
    run = Path(run_directory).resolve()
    output = Path(output_directory).resolve()
    if output.is_relative_to(run):
        raise ValueError("output must be outside the raw run directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"choose an empty output directory: {output}")
    context_path = run / "contexts.json"
    action_path = run / "heavy_action_table.csv"
    contexts = json.loads(context_path.read_text(encoding="utf-8"))
    if not isinstance(contexts, list) or not contexts:
        raise ValueError("contexts.json must contain a nonempty list")

    expected_contexts = {}
    groups, sources, source_groups = {}, {}, {}
    shapes = set()
    source_root = (run / "source").resolve()
    for context in contexts:
        key = _context_key(context, "base_kernel_id")
        kernel = key[0]
        # shortcut: repeated input seeds for one shape are rejected; add seed-aware aggregation when needed.
        if key[:-1] in shapes:
            raise ValueError(f"duplicate shape or multiple input seeds for {key[:-1]}")
        shapes.add(key[:-1])
        group = context["split_group"]
        if not isinstance(group, str) or not group.strip():
            raise ValueError(f"missing split group for {kernel}")
        if context["grid_policy"] != "coverage":
            raise ValueError(f"unsupported grid policy for {kernel}")
        source_path = (source_root / context["source_path"]).resolve()
        if not source_path.is_relative_to(source_root):
            raise ValueError(f"source path escapes source directory: {source_path}")
        source = source_path.read_bytes()
        if hashlib.sha256(source).hexdigest() != context["source_sha256"]:
            raise ValueError(f"source hash mismatch for {kernel}")
        digest = context["source_sha256"]
        if digest in source_groups and source_groups[digest] != group:
            raise ValueError(f"identical sources have different split groups: {kernel}")
        source_groups[digest] = group
        source_info = (context["source_path"], context["source_sha256"], source)
        if kernel in groups and (groups[kernel] != group or sources[kernel] != source_info):
            raise ValueError(f"inconsistent source or split group for {kernel}")
        groups[kernel], sources[kernel] = group, source_info
        expected_contexts[key] = context

    coverage = defaultdict(set)
    normalized = []
    required = {"kernel_id", "m", "n", "k", "input_seed_index", "split_group",
                "block_x", "block_y", "median_ms", "valid", "repeats_ok", "repeats_planned", "grid_policy"}
    with action_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"action table missing columns: {sorted(missing)}")
        for number, row in enumerate(reader, start=2):
            try:
                key = _context_key(row, "kernel_id")
                kernel, m, n, k, _ = key
                if key not in expected_contexts or row["split_group"] != groups[kernel]:
                    raise ValueError("action does not match contexts.json")
                threads = int(row["block_x"])
                if threads not in NLP_THREAD_COUNTS or int(row["block_y"]) != 1:
                    raise ValueError("expected one of the sixteen 1D actions")
                if threads in coverage[key]:
                    raise ValueError("duplicate action")
                runtime_us = float(row["median_ms"]) * 1000.0
                if not math.isfinite(runtime_us) or runtime_us <= 0:
                    raise ValueError("runtime must be finite and positive")
                repeats = int(row["repeats_ok"])
                if row["valid"].lower() != "true" or repeats < 1 or repeats != int(row["repeats_planned"]):
                    raise ValueError("action must be valid with all planned repeats")
                if row["grid_policy"] != "coverage":
                    raise ValueError("unsupported grid policy")
                coverage[key].add(threads)
                normalized.append({
                    "kernel_id": kernel, "m": m, "n": n, "k": k,
                    "block_x": threads, "block_y": 1, "block_z": 1,
                    "runtime_us": runtime_us, "split_group": groups[kernel],
                })
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"invalid heavy action row {number}: {exc}") from exc
    incomplete = [key for key in expected_contexts if coverage[key] != set(NLP_THREAD_COUNTS)]
    if incomplete:
        raise ValueError(f"incomplete 16-action contexts: {incomplete[:5]}")

    cuda = output / "cuda"
    cuda.mkdir(parents=True, exist_ok=True)
    (cuda / "includes.h").write_text(
        "#pragma once\n#include <cuda_runtime.h>\nusing uint = unsigned int;\n", encoding="utf-8"
    )
    for kernel, (source_path, _, source) in sorted(sources.items()):
        if Path(source_path).suffix == ".cuh":
            (cuda / f"{kernel}.cuh").write_bytes(source)
            (cuda / f"{kernel}.cu").write_text(
                f'#include "includes.h"\n#include "{kernel}.cuh"\n', encoding="utf-8"
            )
        else:
            (cuda / f"{kernel}.cu").write_bytes(source)
    fields = ("kernel_id", "m", "n", "k", "block_x", "block_y", "block_z", "runtime_us", "split_group")
    with (output / "runtimes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(sorted(normalized, key=lambda row: tuple(row[name] for name in fields[:5])))
    (output / "split_groups.json").write_text(
        json.dumps(groups, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "source_run": str(run), "problem_columns": ["m", "n", "k"],
        "kernels": len(sources), "contexts": len(expected_contexts),
        "actions_per_context": len(NLP_THREAD_COUNTS), "runtime_rows": len(normalized),
        "split_group_count": len(set(groups.values())),
        "action_table_sha256": hashlib.sha256(action_path.read_bytes()).hexdigest(),
        "contexts_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
        "sources": {kernel: {"path": path, "sha256": digest}
                    for kernel, (path, digest, _) in sorted(sources.items())},
    }
    (output / "dataset.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
