import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from train_sequence import input_fingerprints, load_datasets, make_loader


def build_model(matrix, context_size, action_count, architecture, lstm_hidden_size=128):
    from model.config_scorer import build_model as build_configuration_model

    return build_configuration_model(matrix, context_size, action_count, architecture, lstm_hidden_size)


def dataset_device(run_dir, runtimes_path):
    if not run_dir:
        return None
    run = Path(run_dir)
    metadata_path = run / "metadata.json"
    device = json.loads(metadata_path.read_text(encoding="utf-8"))["device"]

    return {"gpu_name": device["gpu_name"], "compute_capability": device.get("compute_capability"),
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest()}

KL_EMMA_DECAY = 0.99

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtimes", required=True)
    parser.add_argument("--token-ids")
    parser.add_argument("--embedding-matrix")
    parser.add_argument("--checkpoint")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument(
        "--architecture", default="adaptive",
    )
    parser.add_argument("--lstm-hidden-size", type=int, default=1)
    parser.add_argument("--length-penalty", type=float, default=1e-4)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--loss", default="bce")
    parser.add_argument("--seed", type=int, default=2)
    
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--validation-kernels", type=int, default=2)
    parser.add_argument("--problem-columns", nargs="+", default=["problem_size"])
    parser.add_argument("--split-groups")
    parser.add_argument("--validation-groups", type=int, default=2)
    parser.add_argument("--training-groups", type=int)
    parser.add_argument("--dataset-run-dir")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--wandb-mode", default="offline")
    args = parser.parse_args()

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

    catalog, split, training, validation = load_datasets(
        args.runtimes, args.token_ids, matrix, 12.0,
        validation_kernels=args.validation_kernels, problem_columns=args.problem_columns,
        split_groups=groups, validation_groups=args.validation_groups,
        training_groups=args.training_groups,
    )
    source_device = dataset_device(args.dataset_run_dir, args.runtimes)
    model = build_model(matrix, len(training[0].problem_size), len(catalog), args.architecture,
                        lstm_hidden_size=args.lstm_hidden_size)
    coefficient = args.length_penalty if args.architecture == "adaptive" else 0.0
    config = TrainerConfig(epochs=args.epochs, loss=args.loss, device=args.device)
    trainer = AdaptiveTrainer(
        model,
        config,
        length_penalty=coefficient,
        kl_weight=args.kl_weight,
        kl_temperature=2.0,
        kl_ema_decay=KL_EMMA_DECAY,
    )
    train_loader = make_loader(training, args.batch_size, shuffle=True, seed=args.seed)
    validation_loader = make_loader(validation, args.batch_size)
    checkpoint = Path(args.checkpoint or f"checkpoints/{args.architecture}.pt")
    metadata = {
        "format_version": 1, "adaptive_experiment_version": 1, "architecture": args.architecture,
        "lstm_hidden_size": args.lstm_hidden_size,
        "context_size": len(training[0].problem_size), "problem_columns": args.problem_columns,
        "split_groups": groups, "split": {name: list(ids) for name, ids in split.items()},
        "action_catalog": [asdict(action) for action in catalog], "target_exponent": 12.0,
        "input_fingerprints": input_fingerprints(args.runtimes, args.token_ids),
        "embedding_sha256": (
            hashlib.sha256(Path(args.embedding_matrix).read_bytes()).hexdigest()
            if args.embedding_matrix else None
        ),
        "seed": args.seed,
        "batch_size": args.batch_size, "length_penalty": coefficient,
        "kl_weight": args.kl_weight, "kl_temperature": 2,
        "kl_ema_decay": KL_EMMA_DECAY,
        "dataset_device": source_device, "training_samples": len(training), "validation_samples": len(validation),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    environment = timing_environment(args.device, source_device)
    
    wandb_clint = None
    if args.wandb_project:
        import wandb
        wandb_clint = wandb.init(
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
        if (best_key is None or rank < best_key):
            best_key, best_epoch = rank, values["epoch"]
            trainer.save_checkpoint(
                best_path, **metadata, history=checkpoint_history,
                checkpoint_selection="validation_regret_then_task_loss", selected_epoch=best_epoch,
            )
        if wandb_clint:
            metrics = wandb_epoch_metrics(values)
            metrics.update({f"compression/{key}": value for key, value in values.items()
                            if "_h" in key or key.endswith("task_loss")})
            metrics["compression/validation_gate_open_fraction"] = values[
                "validation_gate_open_fraction"
            ]
            metrics["train/kl_loss"] = values["training_kl_loss"]
            metrics["train/weighted_kl_loss"] = args.kl_weight * values["training_kl_loss"]
            metrics["retention/validation_nontrivial_variable_length"] = values["validation_nontrivial_variable_length"]
            wandb_clint.log(metrics, step=values["epoch"])

    
    history = trainer.fit(train_loader, validation_loader, epoch_callback=log_epoch)
    trainer.save_checkpoint(checkpoint, **metadata, history=history,
                                checkpoint_selection="final", selected_epoch=args.epochs)
    cases = prediction_cases(
            model, validation_loader, catalog, args.device, loss_kind=args.loss, length_penalty=coefficient,
            warmup=3, repetitions=10, epoch=args.epochs,
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
                  "timing_environment": environment, "timing_repetitions": 10,
                  "timing_warmup": 3, "compression": compression,
                  "validation": summaries, "cases": cases, "retention": retention,
                  "retention_persistence": persistence,
                  "best_checkpoint": str(best_path),
                  "best_epoch": best_epoch}
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if wandb_clint:
        wandb_clint.summary.update({"best/epoch": best_epoch,
                                "retention/persistent_nontrivial_variation": persistence["persistent_nontrivial_variation"]})
        if retention:
            wandb_clint.summary.update({
                    "retention/nontrivial_variable_length": retention["lengths"]["nontrivial_variable_length"],
                    "retention/extra_state_task_gain": retention["extra_state_task_gain"],
                    "retention/extra_state_regret_gain": retention["extra_state_regret_gain"],
                })
        wandb_clint.summary.update({**summaries, **{f"compression/{k}": compression[k]
                               for k in ("all_same_h", "mean_h", "min_h", "max_h")}})
        wandb_clint.log({"validation/cases": wandb.Table(columns=list(cases[0]),
                     data=[list(row.values()) for row in cases]),
                     "compression/lengths": wandb.Table(columns=list(train_lengths[0]),
                     data=[list(row.values()) for row in [*train_lengths,
                           *[{key: row[key] for key in train_lengths[0]} for row in cases]]])},
                    step=args.epochs + 1)
    if wandb_clint:
        wandb_clint.finish()
        print({"checkpoint": str(checkpoint), "report": str(report_path),
                      "lstm_hidden_size": args.lstm_hidden_size,
                      "best_checkpoint": str(best_path),
                      "best_epoch": best_epoch,
                      "lengths_csv": str(lengths_path), "cases_csv": str(cases_path),
                      "final_epoch": history[-1], "validation": summaries,
                      "compression": compression, "retention": retention,
                      "retention_persistence": persistence, "timing_environment": environment})
    


if __name__ == "__main__":
    main()
