#!/usr/bin/env python3
"""Collect checked LS-CAT launch measurements; Python standard library only."""
import csv
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import secrets
import subprocess
import sys
import time

from analyze import analyze

SIZES = [240, 496, 784, 1016, 1232, 1680, 2024]
PROFILES = {"smoke": ([240], 1), "quick": ([240, 784, 2024], 3), "full": (SIZES, 10)}
SOURCE_FILES = {"vector_add": "10047/0/add.cu", "segmented_sum": "10040/1/gpuSum.cu",
                "periodic_tile": "10073/223/kTile.cu", "repeat_vector": "1018/5/times.cu",
                "matrix_transpose": "10194/34/flatten_kernel.cu", "matrix_add": "10273/0/sumMat.cu"}
FIELDS = ["kernel_id", "repo_id", "source_sha256", "size", "block_x", "block_y",
          "round", "seed", "status", "kernel_ms", "batch_ms", "iterations",
          "grid_x", "grid_y", "grid_z", "input_elements", "output_elements",
          "max_abs_error", "warmup_launches", "elapsed_wall_s", "error", "detail_json"]


def validate_sources(source):
    manifest = json.loads((source / "kernels/manifest.json").read_text())
    ids = set()
    for kernel in manifest["kernels"]:
        name = kernel["id"]
        if not isinstance(name, str) or not name or name in ids:
            raise ValueError("Invalid or duplicate kernel ID in manifest")
        ids.add(name)
        path = (source / "kernels" / kernel["path"]).resolve()
        if not path.is_relative_to((source / "kernels").resolve()):
            raise ValueError("Kernel source path escapes package")
        if SOURCE_FILES.get(name) != kernel["path"]:
            raise ValueError("Kernel ID/path differs from compiled wrapper: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != kernel["sha256"]:
            raise ValueError("Kernel source checksum differs from manifest: " + name)
        if path.with_name("includes.h").read_bytes() != b"":
            raise ValueError("Original includes.h must remain empty: " + name)
    if ids != SOURCE_FILES.keys():
        raise ValueError("Manifest must describe all six compiled kernels")
    return manifest


def make_plan(manifest, device, mode, seed):
    sizes, rounds = PROFILES[mode]
    plan = []
    for kernel in manifest["kernels"]:
        name = kernel["id"]
        resource = device["kernels"][name]
        limit = min(device["max_threads_per_block"], resource["max_threads_per_block"])
        # Native 1D indexing needs (b,1); the historical (1,b) would repeat writes.
        shapes = ([(1, b) for b in range(64, 1025, 64)] + [(b, b) for b in (8, 16, 24, 32)]
                  if name == "matrix_add" else [(b, 1) for b in range(64, 1025, 64)])
        shapes = [(x, y) for x, y in shapes if x * y <= limit
                  and x <= device["max_block_dim_x"] and y <= device["max_block_dim_y"]]
        if not shapes:
            raise ValueError("No supported launch configurations for " + name)
        for size in sizes:
            context_seed = seed ^ int.from_bytes(hashlib.sha256((name + str(size)).encode()).digest()[:4], "little")
            for repetition in range(rounds):
                for x, y in shapes:
                    plan.append({"kernel_id": name, "size": size, "block_x": x,
                                 "block_y": y, "round": repetition, "seed": context_seed})
    random.Random(seed).shuffle(plan)
    return plan


def finite_number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Invalid " + name)
    if positive and value <= 0:
        raise ValueError("Nonpositive " + name)
    return value


def measure(binary, trial, device, timeout=60):
    args = [str(binary), "--device", str(device), "--kernel", trial["kernel_id"],
            "--size", str(trial["size"]), "--block-x", str(trial["block_x"]),
            "--block-y", str(trial["block_y"]), "--seed", str(trial["seed"])]
    row = dict(trial)
    start = time.monotonic()
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
        if proc.returncode:
            row.update(status="process_error", error=(proc.stderr or proc.stdout or "exit " + str(proc.returncode))[-4000:])
        else:
            try:
                result = json.loads(proc.stdout)
                if not isinstance(result, dict) or result.get("status") != "ok" or result.get("correctness") != "passed":
                    raise ValueError("Benchmark did not certify correctness")
                for key in ("kernel_ms", "batch_ms"):
                    finite_number(result.get(key), key, positive=True)
                for key in ("iterations", "grid_x", "grid_y", "grid_z", "input_elements", "output_elements", "warmup_launches"):
                    value = result.get(key)
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        raise ValueError("Invalid " + key)
                finite_number(result.get("max_abs_error"), "max_abs_error")
                if result["max_abs_error"] < 0:
                    raise ValueError("Negative max_abs_error")
                if not math.isclose(result["kernel_ms"] * result["iterations"], result["batch_ms"], rel_tol=1e-5):
                    raise ValueError("Batch timing does not match per-launch timing")
                row.update({key: result[key] for key in FIELDS if key in result and key not in trial})
                row.update(status="ok", detail_json=json.dumps(result, separators=(",", ":")))
            except (ValueError, TypeError, KeyError) as exc:
                row.update(status="protocol_error", error=str(exc) + "; stdout=" + proc.stdout[-3000:])
    except subprocess.TimeoutExpired:
        row.update(status="timeout", error="Benchmark exceeded " + str(timeout) + " seconds")
    row["elapsed_wall_s"] = time.monotonic() - start
    return row


def collect(root, mode, device_index):
    source = root / "source"
    manifest = validate_sources(source)
    binary = root / "benchmark"
    proc = subprocess.run([str(binary), "--describe", "--device", str(device_index)],
                          capture_output=True, text=True, timeout=30, check=False)
    if proc.returncode:
        raise RuntimeError("CUDA device description failed: " + proc.stderr + proc.stdout)
    device = json.loads(proc.stdout)
    seed = secrets.randbits(63)
    plan = make_plan(manifest, device, mode, seed)
    metadata = {"started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "mode": mode, "seed": seed, "cuda_logical_device": device_index,
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID"),
                "dataset_repository": manifest["repository"], "dataset_commit": manifest["commit"],
                "device": device, "planned_trials": len(plan), "timeout_seconds_per_trial": 60,
                "method": "CUDA events, adaptive batch approximately 10ms capped at 1000 launches, full output check",
                "timing_scope": "steady repeated kernel launches; allocation, copies, CPU checks excluded",
                "launch_policy": "native (b,1) for 1D indexing; historical 20 shapes for matrix_add"}
    for name, value in (("metadata.json", metadata), ("planned.json", plan)):
        with (root / name).open("x") as out:
            json.dump(value, out, indent=2)
            out.write("\n")
    by_id = {kernel["id"]: kernel for kernel in manifest["kernels"]}
    completed = 0
    print("Planned checked measurements: " + str(len(plan)), flush=True)
    try:
        with (root / "trials.csv").open("x", newline="") as out:
            writer = csv.DictWriter(out, FIELDS, extrasaction="ignore")
            writer.writeheader()
            out.flush()
            for number, trial in enumerate(plan, 1):
                row = measure(binary, trial, device_index)
                kernel = by_id[trial["kernel_id"]]
                row.update(repo_id=kernel["repo_id"], source_sha256=kernel["sha256"])
                writer.writerow(row)
                out.flush()
                completed = number
                print("{}/{} {} n={} block=({},{}): {}{}".format(
                    number, len(plan), trial["kernel_id"], trial["size"], trial["block_x"],
                    trial["block_y"], row["status"],
                    " {:.6g} ms".format(row["kernel_ms"]) if row["status"] == "ok" else " " + row.get("error", "")[:180]), flush=True)
    finally:
        summary = analyze(root)
        print("Summary: " + str(root / "summary.md"), flush=True)
    return 0 if completed == len(plan) and summary["status"] == "COMPLETE" else 1


def main():
    if len(sys.argv) != 4 or sys.argv[2] not in PROFILES or not sys.argv[3].isdigit():
        print("Usage: python3 collect.py RUN_DIR smoke|quick|full CUDA_LOGICAL_DEVICE", file=sys.stderr)
        return 2
    try:
        return collect(Path(sys.argv[1]).resolve(), sys.argv[2], int(sys.argv[3]))
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as exc:
        print("Collection failed: " + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted; partial results were retained.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
