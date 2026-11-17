"""Evidence for nontrivial learned retention, kept separate from task quality."""

from collections import defaultdict
from dataclasses import asdict
import statistics

import torch

from launch.configuration_catalog import Configuration
from training.adaptive_evaluation import _cached_samples, _length_row
from training.losses import PerformanceTargetLoss
from training.metrics import runtime_metrics
from training.sequence_trainer import ExpectedRuntimeRegretLoss


def length_evidence(rows):
    kernels = {}
    for row in rows:
        kernel = row["kernel_id"]
        item = {name: int(row[name]) for name in ("learned_h", "input_token_count", "hidden_width")}
        if not 1 <= item["learned_h"] <= item["input_token_count"] or item["hidden_width"] < 1:
            raise ValueError("invalid retained length or hidden width")
        if kernel in kernels and kernels[kernel] != item:
            raise ValueError(f"kernel-only representation changed across contexts: {kernel}")
        kernels[kernel] = item
    if not kernels:
        raise ValueError("no kernel representations")
    lengths = sorted({row["learned_h"] for row in kernels.values()})
    partial = sum(1 < row["learned_h"] < row["input_token_count"] for row in kernels.values())
    all_full = all(r["learned_h"] == r["input_token_count"] for r in kernels.values())
    by_input_length = defaultdict(list)
    for row in kernels.values():
        by_input_length[row["input_token_count"]].append(row["learned_h"])
    equal_length_groups = [counts for counts in by_input_length.values() if len(counts) > 1]
    return {
        "kernel_count": len(kernels), "per_kernel": kernels,
        "distinct_h": lengths, "distinct_h_count": len(lengths),
        "variable_output_observed": len(lengths) > 1,
        "all_single_state": lengths == [1],
        "all_full_length": all_full,
        "partially_compressed_kernel_count": partial,
        # Exclude both constant-size output and simply retaining every input token.
        "nontrivial_variable_length": len(lengths) > 1 and not all_full,
        "equal_input_length_different_h": (
            any(len(set(counts)) > 1 for counts in equal_length_groups) if equal_length_groups else None
        ),
        "mean_hidden_values": statistics.fmean(
            r["learned_h"] * r["hidden_width"] for r in kernels.values()
        ),
    }


def persistent_length_evidence(rows, *, through_epoch=None, window=5):
    if window < 1:
        raise ValueError("window must be positive")
    by_epoch = defaultdict(list)
    for row in rows:
        epoch = int(row["epoch"])
        if row["split"] == "validation" and (through_epoch is None or epoch <= through_epoch):
            by_epoch[epoch].append(row)
    end = through_epoch if through_epoch is not None else max(by_epoch, default=0)
    epochs = list(range(max(1, end - window + 1), end + 1))
    complete = len(epochs) == window and all(epoch in by_epoch for epoch in epochs)
    evidence = [length_evidence(by_epoch[epoch]) for epoch in epochs if epoch in by_epoch]
    counts = [{kernel: r["learned_h"] for kernel, r in e["per_kernel"].items()} for e in evidence]
    return {
        "window": window, "through_epoch": end, "window_complete": complete,
        "persistent_nontrivial_variation": complete and all(e["nontrivial_variable_length"] for e in evidence),
        "stable_kernel_counts": complete and all(c == counts[0] for c in counts),
    }


@torch.no_grad()
def retention_diagnostics(model, batches, catalog, device, *, loss_kind="bce", epoch=0):
    """Compare learned retention with only its mandatory final state; no retraining."""
    if not hasattr(model, "gate_network"):
        return None
    model.eval()
    criterion = ExpectedRuntimeRegretLoss() if loss_kind == "regret" else PerformanceTargetLoss(loss_kind)
    rows, selected_logits, final_logits, runtimes_all = [], [], [], []
    for cached, batch, index in _cached_samples(model, batches, device):
        context = batch["context"][index:index + 1].to(device)
        logits = model.score_cached(cached, context)
        last = model.score_cached(cached.final_state_only(), context)
        labels = batch["runtimes" if loss_kind == "regret" else "targets"][index:index + 1].to(device)
        row = _length_row(cached, batch, index, "validation", epoch)
        row.update({
            "selected_task_loss": float(criterion(logits, labels)),
            "final_only_task_loss": float(criterion(last, labels)),
            "selected_threads": catalog.from_id(int(logits.argmax())).threads,
            "final_only_threads": catalog.from_id(int(last.argmax())).threads,
        })
        rows.append(row)
        selected_logits.append(logits.cpu())
        final_logits.append(last.cpu())
        runtimes_all.append(batch["runtimes"][index:index + 1].cpu())
    if not rows:
        raise ValueError("no validation cases")
    runtimes = torch.cat(runtimes_all)
    selected = asdict(runtime_metrics(torch.cat(selected_logits), runtimes))
    final_only = asdict(runtime_metrics(torch.cat(final_logits), runtimes))
    selected["task_loss"] = statistics.fmean(r["selected_task_loss"] for r in rows)
    final_only["task_loss"] = statistics.fmean(r["final_only_task_loss"] for r in rows)
    task_gain = final_only["task_loss"] - selected["task_loss"]
    regret_gain = final_only["mean_regret"] - selected["mean_regret"]
    fixed = None
    try:
        action = catalog.to_id(Configuration(256))
    except KeyError:
        pass
    else:
        logits = torch.zeros_like(runtimes)
        logits[:, action] = 1
        fixed = asdict(runtime_metrics(logits, runtimes))
    return {
        "lengths": length_evidence(rows), "selected": selected, "final_state_only": final_only,
        "extra_state_task_gain": task_gain, "extra_state_regret_gain": regret_gain,
        # Numeric tolerance only, not a statistical significance test.
        "extra_states_improve_task_loss": task_gain > 1e-6,
        "extra_states_improve_regret": regret_gain > 1e-6,
        "fixed_256": fixed, "cases": rows,
    }
