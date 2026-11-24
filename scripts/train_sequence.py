import argparse
import csv
from dataclasses import asdict
from functools import partial
import hashlib
import json
import math
from pathlib import Path


def input_fingerprints(runtimes, token_ids=None) -> dict[str, str]:
    paths = [("runtimes", runtimes)]
    if token_ids is not None:
        paths.append(("token_ids", token_ids))
    return {
        name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for name, path in paths
    }


def build_model(matrix, context_size: int, action_count: int, architecture: str,
                lstm_hidden_size: int = 128):
    from model.config_scorer import build_model as build_configuration_model

    if architecture not in {"baseline", "sequence", "final", "context"}:
        raise ValueError(f"unknown architecture: {architecture}")
    return build_configuration_model(matrix, context_size, action_count, architecture, lstm_hidden_size)


def load_datasets(runtime_path, token_path, matrix, exponent, *, split=None, seed=0,
                  validation_kernels=2, problem_columns=("problem_size",),
                  split_groups=None, validation_groups=2, training_groups=None):
    import torch

    from data.dataset import KernelRuntimeDataset
    from data.runtime_table import RuntimeTable
    from data.splits import split_grouped_kernel_ids, split_train_validation_kernel_ids
    from launch.configuration_catalog import ConfigurationCatalog

    catalog = ConfigurationCatalog.lscat()
    with Path(runtime_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "split_group" in (reader.fieldnames or ()):
            recorded_groups = {}
            for row in reader:
                kernel, group = row["kernel_id"], row["split_group"]
                recorded_groups[kernel] = group
    table = RuntimeTable.from_csv(runtime_path, problem_columns=problem_columns)
    contexts = table.contexts()


    kernel_ids = {key.kernel_id for key in contexts}
    if token_path is None:
        tokens = {kernel_id: [1] for kernel_id in kernel_ids}
    else:
        tokens = json.loads(Path(token_path).read_text(encoding="utf-8"))
    for kernel_id in sorted(kernel_ids):
        sequence = tokens.get(kernel_id)
    # for key in contexts:
    #     pass
    if split is None:
        if split_groups is None:
            split = asdict(split_train_validation_kernel_ids(
                kernel_ids, seed=seed, validation_kernels=validation_kernels
            ))
        else:
            split = asdict(split_grouped_kernel_ids(
                kernel_ids, split_groups, seed=seed, validation_groups=validation_groups,
                training_groups=training_groups,
            ))
    train_ids, validation_ids = set(split["train"]), set(split["validation"])
    selected_ids = train_ids | validation_ids
    
    kernel_ids = selected_ids
    contexts = tuple(key for key in contexts if key.kernel_id in kernel_ids)

    def dataset(ids):
        return KernelRuntimeDataset(
            table, catalog, tokens, kernel_ids=ids, target_exponent=exponent
        )

    train_data, validation_data = dataset(train_ids), dataset(validation_ids)

    return catalog, split, train_data, validation_data


def make_loader(dataset, batch_size, *, shuffle=False, seed=0):
    import torch
    from torch.utils.data import DataLoader
    from data.dataset import collate_samples

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=partial(collate_samples, padding_id=0),
        generator=torch.Generator().manual_seed(seed),
    )


# def parse_args():
#     parser = argparse.ArgumentParser(description=__doc__)
#     parser.add_argument("--runtimes", required=True)
#     parser.add_argument("--token-ids")
#     parser.add_argument("--embedding-matrix")
#     parser.add_argument("--checkpoint", help="default: checkpoints/<architecture>.pt")
#     parser.add_argument(
#         "--architecture", choices=("sequence", "final", "context", "baseline"),
#         default="sequence"
#     )
#     parser.add_argument("--lstm-hidden-size", type=int, default=128,
#                         help="features per LSTM timestep; independent of retained sequence length")
#     parser.add_argument("--epochs", type=int, default=20)
#     parser.add_argument("--batch-size", type=int, default=256)
#     parser.add_argument("--loss", choices=("bce", "mse", "regret"), default="bce")
#     parser.add_argument("--target-exponent", type=float, default=12.0)
#     parser.add_argument("--seed", type=int, default=0)
#     parser.add_argument("--device", default="cpu")
#     parser.add_argument("--validation-kernels", type=int, default=2)
#     parser.add_argument("--problem-columns", nargs="+", default=["problem_size"])
#     parser.add_argument("--split-groups", help="JSON mapping of base kernel IDs to source groups")
#     parser.add_argument("--validation-groups", type=int, default=2,
#                         help="complete groups to reserve when --split-groups is supplied")
#     parser.add_argument("--training-groups", type=int,
#                         help="limit training to this many source groups")
#     parser.add_argument("--verbose", action="store_true")
#     parser.add_argument("--latency-warmup", type=int, default=3)
#     parser.add_argument("--latency-repetitions", type=int, default=5)
#     parser.add_argument("--wandb-project")
#     parser.add_argument("--wandb-entity")
#     parser.add_argument("--wandb-run-name")
#     parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
#     args = parser.parse_args()

#     return args


# def main():
#     args = parse_args()
#     import torch
#     from training.sequence_trainer import SequenceTrainer
#     from training.supervised_trainer import TrainerConfig
#     from training.tracking import WandbEpochLogger, wandb_inference_metrics

#     torch.manual_seed(args.seed)
#     matrix = (
#         torch.zeros((2, 1), dtype=torch.float32)
#         if args.architecture == "context" and not args.embedding_matrix
#         else torch.load(args.embedding_matrix, map_location="cpu", weights_only=True)
#     )
#     split_groups = (
#         json.loads(Path(args.split_groups).read_text(encoding="utf-8"))
#         if args.split_groups else None
#     )
#     if args.split_groups and not isinstance(split_groups, dict):
#         raise ValueError("split-groups JSON must contain a kernel-to-group mapping")
#     catalog, split, train_data, validation_data = load_datasets(
#         args.runtimes, args.token_ids, matrix, args.target_exponent,
#         seed=args.seed, validation_kernels=args.validation_kernels,
#         problem_columns=args.problem_columns, split_groups=split_groups,
#         validation_groups=args.validation_groups, training_groups=args.training_groups,
#     )
#     context_size = len(train_data[0].problem_size)
#     model = build_model(matrix, context_size, len(catalog), args.architecture,
#                         lstm_hidden_size=args.lstm_hidden_size)
#     config = TrainerConfig(
#         epochs=args.epochs, loss=args.loss, device=args.device, verbose=args.verbose
#     )
#     trainer = SequenceTrainer(model, config)
#     checkpoint = Path(args.checkpoint or f"checkpoints/{args.architecture}.pt")
#     metadata = {
#         "format_version": 1,
#         "architecture": args.architecture,
#         "lstm_hidden_size": args.lstm_hidden_size,
#         "context_size": context_size,
#         "problem_columns": args.problem_columns,
#         "split_groups": split_groups,
#         "action_catalog": [asdict(action) for action in catalog],
#         "target_exponent": args.target_exponent,
#         "split": {name: list(ids) for name, ids in split.items()},
#         "seed": args.seed,
#         "batch_size": args.batch_size,
#         "latency_warmup": args.latency_warmup,
#         "latency_repetitions": args.latency_repetitions,
#         "input_fingerprints": input_fingerprints(args.runtimes, args.token_ids),
#         "embedding_sha256": (
#             hashlib.sha256(Path(args.embedding_matrix).read_bytes()).hexdigest()
#             if args.embedding_matrix else None
#         ),
#         "torch_version": str(torch.__version__),
#         "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
#         "training_samples": len(train_data),
#         "validation_samples": len(validation_data),
#     }
#     if args.verbose:
#         print(f"[training:model] {args.architecture}; hidden_size={args.lstm_hidden_size}; "
#               f"{metadata['trainable_parameters']} parameters")
#         print(f"[training:split] {metadata['split']}")
#         print(f"[training:data] train={len(train_data)} validation={len(validation_data)}")
#     train_loader = make_loader(train_data, args.batch_size, shuffle=True, seed=args.seed)
#     validation_loader = make_loader(validation_data, args.batch_size)
#     run = None
#     if args.wandb_project:
#         import wandb
#         run = wandb.init(
#             project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
#             name=args.wandb_run_name or f"{args.architecture}-h{args.lstm_hidden_size}-{args.loss}-seed{args.seed}",
#             config={**vars(args), **metadata, "trainer_config": asdict(config)},
#         )
#     try:
#         history = trainer.fit(
#             train_loader, validation_loader,
#             epoch_callback=WandbEpochLogger(run) if run else None,
#         )
#         latency = trainer.benchmark_inference(
#             validation_loader, warmup=args.latency_warmup,
#             repetitions=args.latency_repetitions,
#         )
#         latency_metrics = wandb_inference_metrics(latency)
#         metadata["runtime_predictor_latency"] = latency_metrics
#         trainer.save_checkpoint(checkpoint, **metadata, history=history)
#         report = {**metadata, "trainer_config": asdict(config), "history": history}
#         report_path = checkpoint.with_suffix(".metrics.json")
#         report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
#         if run:
#             run.log(latency_metrics, step=history[-1]["epoch"])
#             run.summary.update({f"final/{key}": value for key, value in history[-1].items()})
#             run.summary.update(latency_metrics)
#     finally:
#         if run:
#             run.finish()
#     print(json.dumps({"checkpoint": str(checkpoint), "report": str(report_path),
#                       "lstm_hidden_size": args.lstm_hidden_size,
#                       "split": metadata["split"], "final_epoch": history[-1],
#                       "runtime_predictor_latency": latency_metrics}, indent=2))


if __name__ == "__main__":
    main()