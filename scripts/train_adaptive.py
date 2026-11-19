#!/usr/bin/env python3
"""Train adaptive kernel compression or a matched baseline, with length/runtime reports."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from train_sequence import build_model as build_existing_model
from train_sequence import input_fingerprints, load_datasets, make_loader


def build_model(matrix, context_size, action_count, architecture, lstm_hidden_size=128):
    if architecture == "adaptive":
        from model.adaptive_config_scorer import AdaptiveConfigurationModel
        return AdaptiveConfigurationModel(
            matrix, context_size, action_count, hidden_size=lstm_hidden_size
        )
    return build_existing_model(matrix, context_size, action_count, architecture,
                                lstm_hidden_size=lstm_hidden_size)


def dataset_device(run_dir, runtimes_path):
    if not run_dir:
        return None
    run = Path(run_dir)
    metadata_path = run / "metadata.json"
    device = json.loads(metadata_path.read_text(encoding="utf-8"))["device"]
    if not isinstance(device, dict) or not device.get("gpu_name"):
        raise ValueError("dataset metadata must identify its GPU")
    preparation = Path(runtimes_path).parent / "dataset.json"
    if preparation.is_file():
        prepared = json.loads(preparation.read_text(encoding="utf-8"))
        digest = hashlib.sha256((run / "heavy_action_table.csv").read_bytes()).hexdigest()
        if prepared.get("action_table_sha256") != digest:
            raise ValueError("dataset run does not match the prepared action-table provenance")
    return {"gpu_name": device["gpu_name"], "compute_capability": device.get("compute_capability"),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtimes", required=True)
    parser.add_argument("--token-ids")
    parser.add_argument("--embedding-matrix")
    parser.add_argument("--checkpoint")
    parser.add_argument("--save-best", action="store_true",
                        help="also save .best.pt by validation regret, then task loss")
    parser.add_argument(
        "--architecture", choices=("adaptive", "final", "context", "baseline", "sequence"),
        default="adaptive",
    )
    parser.add_argument("--lstm-hidden-size", type=int, default=128,
                        help="features per LSTM timestep; independent of retained sequence length")
    parser.add_argument("--length-penalty", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--loss", choices=("bce", "mse", "regret"), default="bce")
    parser.add_argument("--target-exponent", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-seed", type=int,
                        help="hold the data split fixed while changing the model seed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validation-kernels", type=int, default=2)
    parser.add_argument("--problem-columns", nargs="+", default=["problem_size"])
    parser.add_argument("--split-groups")
    parser.add_argument("--validation-groups", type=int, default=2)
    parser.add_argument("--training-groups", type=int,
                        help="limit training to this many source groups")
    parser.add_argument("--dataset-run-dir", help="raw run used to identify the dataset GPU")
    parser.add_argument("--latency-warmup", type=int, default=3)
    parser.add_argument("--latency-repetitions", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if args.lstm_hidden_size < 1:
        parser.error("LSTM hidden size must be positive")
    if not math.isfinite(args.length_penalty) or args.length_penalty < 0:
        parser.error("length penalty must be finite and nonnegative")
    if args.latency_warmup < 0 or args.latency_repetitions < 1:
        parser.error("latency warmup must be nonnegative and repetitions positive")
    if args.architecture != "context" and (not args.token_ids or not args.embedding_matrix):
        parser.error("--token-ids and --embedding-matrix are required except for context-only")
    return args


def main():
    args = parse_args()
    import torch
    from training.adaptive_evaluation import (
        case_summary, length_summary, prediction_cases, representation_rows, timing_environment, write_rows,
    )
    from training.adaptive_trainer import AdaptiveTrainer
    from training.retention_evaluation import persistent_length_evidence, retention_diagnostics
    from training.supervised_trainer import TrainerConfig
    from training.tracking import wandb_epoch_metrics

    torch.manual_seed(args.seed)
    matrix = (
        torch.zeros((2, 1), dtype=torch.float32)
        if args.architecture == "context" and not args.embedding_matrix
        else torch.load(args.embedding_matrix, map_location="cpu", weights_only=True)
    )
    groups = json.loads(Path(args.split_groups).read_text(encoding="utf-8")) if args.split_groups else None
    if args.split_groups and not isinstance(groups, dict):
        raise ValueError("split groups must be a JSON kernel-to-group mapping")
    split_seed = args.seed if args.split_seed is None else args.split_seed
    catalog, split, training, validation = load_datasets(
        args.runtimes, args.token_ids, matrix, args.target_exponent, seed=split_seed,
        validation_kernels=args.validation_kernels, problem_columns=args.problem_columns,
        split_groups=groups, validation_groups=args.validation_groups,
        training_groups=args.training_groups,
    )
    source_device = dataset_device(args.dataset_run_dir, args.runtimes)
    model = build_model(matrix, len(training[0].problem_size), len(catalog), args.architecture,
                        lstm_hidden_size=args.lstm_hidden_size)
    coefficient = args.length_penalty if args.architecture == "adaptive" else 0.0
    config = TrainerConfig(epochs=args.epochs, loss=args.loss, device=args.device, verbose=args.verbose)
    trainer = AdaptiveTrainer(model, config, length_penalty=coefficient)
    train_loader = make_loader(training, args.batch_size, shuffle=True, seed=args.seed)
    validation_loader = make_loader(validation, args.batch_size)
    checkpoint = Path(args.checkpoint or f"checkpoints/{args.architecture}.pt")
    metadata = {
        "format_version": 1, "adaptive_experiment_version": 1, "architecture": args.architecture,
        "lstm_hidden_size": args.lstm_hidden_size,
        "context_size": len(training[0].problem_size), "problem_columns": args.problem_columns,
        "split_groups": groups, "split": {name: list(ids) for name, ids in split.items()},
        "action_catalog": [asdict(action) for action in catalog], "target_exponent": args.target_exponent,
        "input_fingerprints": input_fingerprints(args.runtimes, args.token_ids),
        "embedding_sha256": (
            hashlib.sha256(Path(args.embedding_matrix).read_bytes()).hexdigest()
            if args.embedding_matrix else None
        ),
        "seed": args.seed, "split_seed": split_seed,
        "batch_size": args.batch_size, "length_penalty": coefficient,
        "dataset_device": source_device, "training_samples": len(training), "validation_samples": len(validation),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    environment = timing_environment(args.device, source_device)
    if args.verbose:
        print(f"[adaptive:model] {args.architecture} hidden_size={args.lstm_hidden_size} lambda={coefficient:g}")
        print(f"[adaptive:split] {metadata['split']}")
        print(f"[adaptive:timing] {environment}")
    run = None
    if args.wandb_project:
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
            name=args.wandb_run_name or f"{args.architecture}-h{args.lstm_hidden_size}-{args.loss}-lambda{coefficient:g}-seed{args.seed}",
            config={**vars(args), **metadata, "timing_environment": environment},
        )

    best_path = checkpoint.with_suffix(".best.pt")
    best_key, best_epoch = None, None
    checkpoint_history = []

    def log_epoch(values):
        nonlocal best_key, best_epoch
        checkpoint_history.append(dict(values))
        rank = (values["mean_regret"], values["validation_task_loss"])
        if args.save_best and not all(math.isfinite(value) for value in rank):
            raise FloatingPointError("non-finite checkpoint selection metric")
        if args.save_best and (best_key is None or rank < best_key):
            best_key, best_epoch = rank, values["epoch"]
            trainer.save_checkpoint(
                best_path, **metadata, history=checkpoint_history,
                checkpoint_selection="validation_regret_then_task_loss", selected_epoch=best_epoch,
            )
        if run:
            metrics = wandb_epoch_metrics(values)
            metrics.update({f"compression/{key}": value for key, value in values.items()
                            if "_h" in key or key.endswith("task_loss")})
            metrics["compression/validation_gate_open_fraction"] = values[
                "validation_gate_open_fraction"
            ]
            metrics["retention/validation_nontrivial_variable_length"] = values["validation_nontrivial_variable_length"]
            run.log(metrics, step=values["epoch"])

    try:
        history = trainer.fit(train_loader, validation_loader, epoch_callback=log_epoch)
        # Save trained weights even if a later timing/reporting step fails.
        trainer.save_checkpoint(checkpoint, **metadata, history=history,
                                checkpoint_selection="final", selected_epoch=args.epochs)
        cases = prediction_cases(
            model, validation_loader, catalog, args.device, loss_kind=args.loss, length_penalty=coefficient,
            warmup=args.latency_warmup, repetitions=args.latency_repetitions, epoch=args.epochs,
        )
        train_lengths = representation_rows(model, make_loader(training, args.batch_size), args.device,
                                            split="training", epoch=args.epochs)
        compression = length_summary([*train_lengths, *cases])
        summaries = case_summary(cases)
        retention = retention_diagnostics(
            model, validation_loader, catalog, args.device, loss_kind=args.loss, epoch=args.epochs
        )
        persistence = persistent_length_evidence(trainer.length_history)
        lengths_path = checkpoint.with_suffix(".lengths.csv")
        cases_path = checkpoint.with_suffix(".validation.csv")
        report_path = checkpoint.with_suffix(".metrics.json")
        write_rows(lengths_path, [*trainer.length_history, *train_lengths], args.problem_columns)
        write_rows(cases_path, cases, args.problem_columns)
        report = {**metadata, "trainer_config": asdict(config), "history": history,
                  "timing_environment": environment, "timing_repetitions": args.latency_repetitions,
                  "timing_warmup": args.latency_warmup, "compression": compression,
                  "validation": summaries, "cases": cases, "retention": retention,
                  "retention_persistence": persistence,
                  "best_checkpoint": str(best_path) if args.save_best else None,
                  "best_epoch": best_epoch}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if run:
            run.summary.update({"best/epoch": best_epoch,
                                "retention/persistent_nontrivial_variation": persistence["persistent_nontrivial_variation"]})
            if retention:
                run.summary.update({
                    "retention/nontrivial_variable_length": retention["lengths"]["nontrivial_variable_length"],
                    "retention/extra_state_task_gain": retention["extra_state_task_gain"],
                    "retention/extra_state_regret_gain": retention["extra_state_regret_gain"],
                })
            run.summary.update({**summaries, **{f"compression/{k}": compression[k]
                               for k in ("all_same_h", "mean_h", "min_h", "max_h")}})
            run.log({"validation/cases": wandb.Table(columns=list(cases[0]),
                     data=[list(row.values()) for row in cases]),
                     "compression/lengths": wandb.Table(columns=list(train_lengths[0]),
                     data=[list(row.values()) for row in [*train_lengths,
                           *[{key: row[key] for key in train_lengths[0]} for row in cases]]])},
                    step=args.epochs + 1)
    finally:
        if run:
            run.finish()
    print(json.dumps({"checkpoint": str(checkpoint), "report": str(report_path),
                      "lstm_hidden_size": args.lstm_hidden_size,
                      "best_checkpoint": str(best_path) if args.save_best else None,
                      "best_epoch": best_epoch,
                      "lengths_csv": str(lengths_path), "cases_csv": str(cases_path),
                      "final_epoch": history[-1], "validation": summaries,
                      "compression": compression, "retention": retention,
                      "retention_persistence": persistence, "timing_environment": environment}, indent=2))


if __name__ == "__main__":
    main()
