from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

NLP_THREAD_COUNTS = tuple(range(64, 1025, 64))


def _context_key(row, kernel_field):
    kernel = row[kernel_field]
    dimensions = tuple(int(row[name]) for name in ("m", "n", "k"))
    seed_index = int(row["input_seed_index"])
    return kernel, *dimensions, seed_index


def prepare_one_d_dataset(run_directory, output_directory):
    run = Path(run_directory).resolve()
    output = Path(output_directory).resolve()
    context_path = run / "contexts.json"
    action_path = run / "heavy_action_table.csv"
    contexts = json.loads(context_path.read_text(encoding="utf-8"))

    expected_contexts = {}
    groups, sources, source_groups = {}, {}, {}
    shapes = set()
    source_root = (run / "source").resolve()
    for context in contexts:
        key = _context_key(context, "base_kernel_id")
        kernel = key[0]

        shapes.add(key[:-1])
        group = context["split_group"]
        source_path = (source_root / context["source_path"]).resolve()
        source = source_path.read_bytes()
        digest = context["source_sha256"]
        source_groups[digest] = group
        source_info = (context["source_path"], context["source_sha256"], source)
        groups[kernel], sources[kernel] = group, source_info
        expected_contexts[key] = context

    coverage = defaultdict(set)
    normalized = []
    with action_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = _context_key(row, "kernel_id")
            kernel, m, n, k, _ = key
            threads = int(row["block_x"])
            runtime_us = float(row["median_ms"]) * 1000.0
            coverage[key].add(threads)
            normalized.append({
                "kernel_id": kernel, "m": m, "n": n, "k": k,
                "block_x": threads, "block_y": 1, "block_z": 1,
                "runtime_us": runtime_us, "split_group": groups[kernel],
            })

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
