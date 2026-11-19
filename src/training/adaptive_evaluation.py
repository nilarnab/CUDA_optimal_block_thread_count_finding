"""Per-case compression and cached-predictor timing, without launching dataset kernels."""

from __future__ import annotations

from collections import Counter
import csv
import json
from pathlib import Path
import platform
import statistics
import time

import torch

from training.losses import PerformanceTargetLoss
from training.sequence_trainer import ExpectedRuntimeRegretLoss


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _cached_samples(model, batches, device):
    caches = {}
    for batch in batches:
        for index, kernel in enumerate(batch["kernel_ids"]):
            if kernel not in caches:
                length = int(batch["lengths"][index])
                caches[kernel] = model.encode_kernel(
                    batch["tokens"][index:index + 1, :length].to(device),
                    batch["lengths"][index:index + 1].to(device),
                )
            yield caches[kernel], batch, index


def _length_row(cached, batch, index, split, epoch):
    sequence = hasattr(cached, "lengths")
    context_only = (
        isinstance(cached, torch.Tensor)
        and cached.ndim == 2
        and cached.size(-1) == 0
    )
    actual = int(cached.lengths[0]) if sequence else (0 if context_only else 1)
    expected = float(cached.expected_lengths[0]) if hasattr(cached, "expected_lengths") else float(actual)
    width = cached.states.size(-1) if sequence else cached.size(-1)
    return {
        "epoch": epoch, "split": split, "kernel_id": batch["kernel_ids"][index],
        "problem_size": batch["context"][index].tolist(),
        "input_token_count": int(batch["lengths"][index]), "learned_h": actual,
        "expected_h": expected, "hidden_width": width,
    }


@torch.no_grad()
def representation_rows(model, batches, device, *, split="training", epoch=0):
    model.eval()
    return [_length_row(cached, batch, index, split, epoch)
            for cached, batch, index in _cached_samples(model, batches, device)]


def length_summary(rows):
    kernels = {}
    for row in rows:
        kernel = row["kernel_id"]
        if kernel in kernels and kernels[kernel]["learned_h"] != row["learned_h"]:
            raise ValueError(f"kernel-only representation length changed across problem sizes: {kernel}")
        kernels[kernel] = {name: row[name] for name in ("learned_h", "input_token_count", "expected_h")}
    if not kernels:
        raise ValueError("no representations to summarize")
    lengths = [row["learned_h"] for row in kernels.values()]
    return {
        "per_kernel": kernels, "kernel_count": len(kernels),
        "h_histogram": dict(sorted(Counter(map(str, lengths)).items())),
        "all_same_h": len(set(lengths)) == 1, "mean_h": statistics.fmean(lengths),
        "min_h": min(lengths), "max_h": max(lengths),
        "compressed_kernel_count": sum(row["learned_h"] < row["input_token_count"] for row in kernels.values()),
        "mean_retained_fraction": statistics.fmean(
            row["learned_h"] / row["input_token_count"] for row in kernels.values()
        ),
    }


def timing_environment(device, dataset_device=None):
    device = torch.device(device)
    measured_name = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()
    dataset_name = (dataset_device or {}).get("gpu_name")
    return {
        "device": str(device), "predictor_device_name": measured_name,
        "dataset_gpu_name": dataset_name,
        "same_gpu_model": device.type == "cuda" and dataset_name == measured_name if dataset_name else None,
        "torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "dense_timer": "cuda_events" if device.type == "cuda" else "synchronized_host_clock",
        "predictor_timer": "synchronized_host_clock",
        "kernel_timer": "recorded_dataset_median", "actual_kernel_execution": False,
    }


@torch.no_grad()
def prediction_cases(model, batches, catalog, device, *, loss_kind="bce", length_penalty=0.0,
                     warmup=3, repetitions=10, epoch=0):
    if warmup < 0 or repetitions < 1:
        raise ValueError("warmup must be nonnegative and repetitions positive")
    model.eval()
    device = torch.device(device)
    criterion = ExpectedRuntimeRegretLoss() if loss_kind == "regret" else PerformanceTargetLoss(loss_kind)
    rows = []
    for cached, batch, index in _cached_samples(model, batches, device):
        row = _length_row(cached, batch, index, "validation", epoch)
        dimensions = tuple(float(v) for v in batch["context"][index])
        context = torch.tensor([dimensions], dtype=torch.float32, device=device)
        # Capture the dense input once; measurement has no instrumentation hooks.
        inputs = []
        hook = model.scorer.register_forward_pre_hook(lambda module, args: inputs.append(args[0]))
        try:
            logits = model.score_cached(cached, context)
        finally:
            hook.remove()
        dense_input = inputs[0]
        action = int(logits.argmax(dim=-1).item())
        label_key = "runtimes" if loss_kind == "regret" else "targets"
        task_loss = float(criterion(logits, batch[label_key][index:index + 1].to(device)))

        def choose():
            launch_context = torch.tensor([dimensions], dtype=torch.float32, device=device)
            return int(model.score_cached(cached, launch_context).argmax(dim=-1).item())

        for _ in range(warmup):
            model.scorer(dense_input)
            choose()
        _synchronize(device)
        dense_times, predictor_times = [], []
        if device.type == "cuda":
            with torch.cuda.device(device):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                for _ in range(repetitions):
                    start.record()
                    model.scorer(dense_input)
                    end.record()
                    end.synchronize()
                    dense_times.append(start.elapsed_time(end))
        else:
            for _ in range(repetitions):
                _synchronize(device)
                started = time.perf_counter_ns()
                model.scorer(dense_input)
                _synchronize(device)
                dense_times.append((time.perf_counter_ns() - started) / 1e6)
        for _ in range(repetitions):
            _synchronize(device)
            started = time.perf_counter_ns()
            selected = choose()
            _synchronize(device)
            predictor_times.append((time.perf_counter_ns() - started) / 1e6)
            if selected != action:
                raise RuntimeError("cached evaluation produced inconsistent action choices")
        runtimes = batch["runtimes"][index]
        selected_ms = float(runtimes[action]) / 1000.0
        optimal_ms = float(runtimes.min()) / 1000.0
        dense_ms, predictor_ms = statistics.median(dense_times), statistics.median(predictor_times)
        row.update({
            "selected_action": action, "selected_threads": catalog.from_id(action).threads,
            "optimal_action": int(runtimes.argmin()), "selected_kernel_ms": selected_ms,
            "optimal_kernel_ms": optimal_ms, "dense_ms": dense_ms, "predictor_ms": predictor_ms,
            "dense_plus_kernel_ms": dense_ms + selected_ms,
            "estimated_total_ms": predictor_ms + selected_ms,
            "performance_score": optimal_ms / selected_ms, "regret": selected_ms / optimal_ms - 1,
            "task_loss": task_loss, "objective": task_loss + length_penalty * row["expected_h"],
        })
        rows.append(row)
    if not rows:
        raise ValueError("no validation cases")
    return rows


def case_summary(rows):
    result = {"cases": len(rows)}
    for name in ("dense_ms", "predictor_ms", "selected_kernel_ms", "estimated_total_ms", "dense_plus_kernel_ms"):
        result[f"mean_{name}"] = statistics.fmean(row[name] for row in rows)
    result["performance_score"] = statistics.fmean(row["performance_score"] for row in rows)
    result["validation_task_loss"] = statistics.fmean(row["task_loss"] for row in rows)
    result["validation_loss"] = statistics.fmean(row["objective"] for row in rows)
    result["mean_regret"] = statistics.fmean(row["regret"] for row in rows)
    result["exact_optimal_rate"] = statistics.fmean(
        row["selected_action"] == row["optimal_action"] for row in rows
    )
    return result


def write_rows(path, rows, problem_columns):
    if not rows:
        raise ValueError("no rows to write")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = [name for name in problem_columns if name not in rows[0]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*rows[0], *extra_columns])
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "problem_size": json.dumps(row["problem_size"]),
                             **{name: value for name, value in zip(problem_columns, row["problem_size"])
                                if name in extra_columns}})
