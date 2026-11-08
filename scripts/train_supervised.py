#!/usr/bin/env python3
"""Train the generic supervised selector from a normalized runtime CSV.

This script intentionally expects pre-tokenized kernels in a JSON mapping. The
separate frontend can be used to produce LLVM IR and tokens without coupling
compilation to every training run.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import re


def wandb_run_name(args: argparse.Namespace) -> str:
    """Build a readable run name from the experiment hyperparameters."""

    prefix = args.wandb_run_name or "lscat"
    prefix = re.sub(r"[^A-Za-z0-9_.-]+", "_", prefix).strip("_.-") or "lscat"
    return "-".join(
        (
            prefix,
            f"ep{args.epochs}",
            f"bs{args.batch_size}",
            f"loss-{args.loss}",
            f"exp{args.target_exponent:g}",
            f"seed{args.seed}",
            f"dev-{re.sub(r'[^A-Za-z0-9_.-]+', '_', args.device)}",
            f"valk{args.validation_kernels}",
            f"lw{args.latency_warmup}",
            f"lr{args.latency_repetitions}",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtimes", required=True)
    parser.add_argument("--token-ids", required=True)
    parser.add_argument("--embedding-matrix", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/supervised.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--loss", choices=("bce", "mse"), default="bce")
    parser.add_argument("--target-exponent", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--validation-kernels",
        type=int,
        default=2,
        help="number of complete unseen kernels reserved for validation",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--wandb-project",
        help="enable Weights & Biases logging for this project",
    )
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-run-name",
        help="optional run-name prefix; hyperparameters are appended automatically",
    )
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--latency-warmup", type=int, default=3)
    parser.add_argument("--latency-repetitions", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from torch.utils.data import DataLoader

    from data.dataset import KernelRuntimeDataset, collate_samples
    from data.runtime_table import RuntimeTable
    from data.splits import split_train_validation_kernel_ids
    from encoders.context_encoder import ContextEncoder
    from encoders.fasttext_encoder import FrozenFastTextEncoder
    from encoders.lstm_encoder import LSTMEncoder
    from launch.configuration_catalog import ConfigurationCatalog
    from model.config_scorer import ConfigurationModel, ConfigurationScorer
    from training.supervised_trainer import SupervisedTrainer, TrainerConfig
    from training.tracking import WandbEpochLogger, wandb_inference_metrics

    torch.manual_seed(args.seed)
    if args.verbose:
        print(f"[training:input] runtimes: {args.runtimes}")
        print(f"[training:input] token IDs: {args.token_ids}")
        print(f"[training:input] embedding matrix: {args.embedding_matrix}")
        print(f"[training:config] seed={args.seed} device={args.device}")
    catalog = ConfigurationCatalog.lscat()
    runtimes = RuntimeTable.from_csv(args.runtimes)
    token_ids = json.loads(Path(args.token_ids).read_text(encoding="utf-8"))
    split = split_train_validation_kernel_ids(
        (key.kernel_id for key in runtimes.contexts()),
        validation_kernels=args.validation_kernels,
        seed=args.seed,
    )
    if args.verbose:
        print(f"[training:catalog] {len(catalog)} actions: {[c.threads for c in catalog]}")
        print(f"[training:split] training kernels: {list(split.train)}")
        print(f"[training:split] validation kernels: {list(split.validation)}")

    def dataset(kernel_ids):
        return KernelRuntimeDataset(
            runtimes,
            catalog,
            token_ids,
            kernel_ids=kernel_ids,
            target_exponent=args.target_exponent,
        )

    train_data = dataset(split.train)
    validation_data = dataset(split.validation)
    if not train_data or not validation_data:
        raise SystemExit(
            "the kernel-level split produced an empty training or validation set; "
            "provide more distinct kernels"
        )
    if args.verbose:
        print(f"[training:data] training samples: {len(train_data)}")
        print(f"[training:data] validation samples: {len(validation_data)}")
    collate = partial(collate_samples, padding_id=0)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    validation_loader = DataLoader(
        validation_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    matrix = torch.load(args.embedding_matrix, weights_only=True)
    if args.verbose:
        print(f"[training:model] embedding matrix shape: {tuple(matrix.shape)}")
    token_encoder = FrozenFastTextEncoder(matrix)
    kernel_encoder = LSTMEncoder(token_encoder.output_size, 128, attention=True)
    context_size = len(train_data[0].problem_size)
    context_encoder = ContextEncoder(context_size, 32)
    scorer = ConfigurationScorer(
        kernel_encoder.output_size + context_encoder.output_size, len(catalog)
    )
    model = ConfigurationModel(token_encoder, kernel_encoder, context_encoder, scorer)
    if args.verbose:
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        frozen = sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        )
        print(f"[training:model] context dimensions: {context_size}")
        print(f"[training:model] output scores: {len(catalog)}")
        print(f"[training:model] trainable parameters: {trainable}")
        print(f"[training:model] frozen parameters: {frozen}")
    trainer = SupervisedTrainer(
        model,
        TrainerConfig(
            epochs=args.epochs,
            loss=args.loss,
            device=args.device,
            verbose=args.verbose,
        ),
    )
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
        except ImportError as exc:
            raise SystemExit(
                "W&B logging requires 'pip install -e .[tracking]'"
            ) from exc
        generated_run_name = wandb_run_name(args)
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=generated_run_name,
            mode=args.wandb_mode,
            config={
                "runtimes_path": str(Path(args.runtimes).resolve()),
                "token_ids_path": str(Path(args.token_ids).resolve()),
                "embedding_matrix_path": str(
                    Path(args.embedding_matrix).resolve()
                ),
                "checkpoint_path": str(Path(args.checkpoint).resolve()),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "loss": args.loss,
                "target_exponent": args.target_exponent,
                "seed": args.seed,
                "device": args.device,
                "validation_kernel_count": args.validation_kernels,
                "training_kernels": list(split.train),
                "validation_kernels": list(split.validation),
                "training_samples": len(train_data),
                "validation_samples": len(validation_data),
                "actions": [configuration.threads for configuration in catalog],
                "embedding_shape": list(matrix.shape),
                "trainable_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "frozen_parameters": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if not parameter.requires_grad
                ),
                "latency_warmup": args.latency_warmup,
                "latency_repetitions": args.latency_repetitions,
            },
        )
        if args.verbose:
            print(
                f"[training:wandb] project={args.wandb_project} "
                f"mode={args.wandb_mode} run={generated_run_name}"
            )

    try:
        history = trainer.fit(
            train_loader,
            validation_loader,
            epoch_callback=(WandbEpochLogger(wandb_run) if wandb_run else None),
        )
        latency = trainer.benchmark_inference(
            validation_loader,
            warmup=args.latency_warmup,
            repetitions=args.latency_repetitions,
        )
        latency_metrics = wandb_inference_metrics(latency)
        history[-1]["inference_latency"] = latency_metrics
        if args.verbose:
            print(
                "[training:latency] "
                f"mean={latency.mean_ms:.4f}ms "
                f"median={latency.median_ms:.4f}ms "
                f"p95={latency.p95_ms:.4f}ms "
                f"throughput={latency.samples_per_second:.2f} samples/s"
            )
        if wandb_run:
            wandb_run.log(latency_metrics, step=history[-1]["epoch"])
        trainer.save_checkpoint(
            args.checkpoint,
            history=history,
            action_catalog=[configuration.__dict__ for configuration in catalog],
            context_size=context_size,
            target_exponent=args.target_exponent,
            training_kernels=list(split.train),
            validation_kernels=list(split.validation),
        )
        if args.verbose:
            print(f"[training:checkpoint] saved to {args.checkpoint}")
        if wandb_run:
            final = history[-1]
            wandb_run.summary.update(
                {
                    "final/training_loss": final["training_loss"],
                    "final/validation_loss": final["validation_loss"],
                    "final/performance_score": final["performance_score"],
                    "final/mean_regret": final["mean_regret"],
                    "final/exact_optimal_rate": final["exact_optimal_rate"],
                    **latency_metrics,
                    "checkpoint_path": str(Path(args.checkpoint).resolve()),
                }
            )
    finally:
        if wandb_run:
            wandb_run.finish()
    print(
        json.dumps(
            {
                "split": {
                    "training_kernels": list(split.train),
                    "validation_kernels": list(split.validation),
                    "training_samples": len(train_data),
                    "validation_samples": len(validation_data),
                },
                "final_epoch": history[-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

