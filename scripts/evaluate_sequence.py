#!/usr/bin/env python3
"""Reload saved weights and compare runtime choices on the saved validation split."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from train_sequence import build_model, input_fingerprints, load_datasets, make_loader


def evaluate_checkpoint(path, args):
    import torch
    from training.sequence_trainer import SequenceTrainer
    from training.supervised_trainer import TrainerConfig

    saved = torch.load(path, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]
    if metadata.get("format_version") != 1 or "architecture" not in metadata:
        raise ValueError("use a checkpoint produced by scripts/train_sequence.py")
    if metadata["input_fingerprints"] != input_fingerprints(args.runtimes, args.token_ids):
        raise ValueError("runtime/token files differ from the checkpoint's training inputs")
    matrix = saved["model_state"]["token_encoder.embedding.weight"]
    catalog, split, _, validation = load_datasets(
        args.runtimes, args.token_ids, matrix, metadata["target_exponent"], split=metadata["split"]
    )
    if metadata["action_catalog"] != [asdict(action) for action in catalog]:
        raise ValueError("action catalog differs from the checkpoint")
    model = build_model(matrix, metadata["context_size"], len(catalog), metadata["architecture"])
    model.load_state_dict(saved["model_state"])
    config = TrainerConfig(**{**saved["trainer_config"], "device": args.device})
    trainer = SequenceTrainer(model, config)
    loss, metrics = trainer.evaluate(make_loader(validation, args.batch_size))

    selections = []
    with torch.no_grad():
        for batch in make_loader(validation, args.batch_size):
            logits = model(
                batch["tokens"].to(args.device), batch["lengths"].to(args.device),
                batch["context"].to(args.device),
            )
            actions = logits.argmax(dim=-1).cpu()
            for index, action in enumerate(actions.tolist()):
                runtimes = batch["runtimes"][index]
                best = float(runtimes.min())
                selected = float(runtimes[action])
                selections.append({
                    "kernel_id": batch["kernel_ids"][index],
                    "problem_size": batch["context"][index].tolist(),
                    "selected_threads": catalog.from_id(action).threads,
                    "optimal_threads": catalog.from_id(int(runtimes.argmin())).threads,
                    "selected_runtime_us": selected,
                    "optimal_runtime_us": best,
                    "performance_score": best / selected,
                    "regret": selected / best - 1,
                })
    return {
        "checkpoint": str(path), "architecture": metadata["architecture"],
        "loss_kind": config.loss, "target_exponent": metadata["target_exponent"],
        "validation_loss": loss, **asdict(metrics),
        "validation_kernels": list(split["validation"]),
        "trainable_parameters": metadata["trainable_parameters"],
        "selections": selections,
    }, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compare", help="baseline checkpoint from train_sequence.py")
    parser.add_argument("--runtimes", required=True)
    parser.add_argument("--token-ids", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", help="also save the JSON report here")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch size must be positive")
    result, metadata = evaluate_checkpoint(args.checkpoint, args)
    report = {"model": result}
    if args.compare:
        baseline, baseline_metadata = evaluate_checkpoint(args.compare, args)
        for field in ("split", "input_fingerprints", "embedding_sha256", "action_catalog"):
            if metadata[field] != baseline_metadata[field]:
                raise ValueError(f"cannot compare checkpoints with different {field}")
        report["comparison"] = baseline
        report["delta_model_minus_comparison"] = {
            name: result[name] - baseline[name]
            for name in ("performance_score", "mean_regret", "exact_optimal_rate")
        }
        baseline_time = sum(row["selected_runtime_us"] for row in baseline["selections"])
        model_time = sum(row["selected_runtime_us"] for row in result["selections"])
        report["selected_runtime_sum_speedup"] = baseline_time / model_time
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
