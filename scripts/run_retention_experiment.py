#!/usr/bin/env python3
"""Matched width-8 controls and an honest signal for useful variable retention."""

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/processed_1d")
    parser.add_argument("--dataset-run-dir")
    parser.add_argument("--output-dir", default="checkpoints/retention_h8_experiment")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument("--validation-groups", type=int, default=2)
    parser.add_argument("--length-penalties", nargs="+", type=float, default=[0.0, 1e-5, 1e-4])
    parser.add_argument("--lstm-hidden-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--loss", choices=("bce", "mse", "regret"), default="bce")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--latency-warmup", type=int, default=3)
    parser.add_argument("--latency-repetitions", type=int, default=10)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.lstm_hidden_size, args.validation_groups, args.latency_repetitions) < 1:
        parser.error("epochs, batch size, width, group count, and repetitions must be positive")
    if args.latency_warmup < 0:
        parser.error("warmup must be nonnegative")
    if any(not math.isfinite(p) or p < 0 for p in args.length_penalties):
        parser.error("length penalties must be finite and nonnegative")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.length_penalties)) != len(args.length_penalties):
        parser.error("seeds and penalties must not contain duplicates")
    return args


def retention_signal(diagnostics, persistence):
    if diagnostics is None:
        return {"variable_output_signal": False, "extra_states_help_signal": False}
    variable = (diagnostics["lengths"]["nontrivial_variable_length"]
                and persistence["persistent_nontrivial_variation"])
    return {
        "variable_output_signal": variable,
        "extra_states_help_signal": variable and diagnostics["extra_states_improve_regret"],
    }


def main():
    args = parse_args()
    from training.retention_evaluation import persistent_length_evidence

    data, output = Path(args.data_dir).resolve(), Path(args.output_dir).resolve()
    for name in ("runtimes.csv", "token_ids.json", "fasttext_embeddings.pt", "split_groups.json"):
        if not (data / name).is_file():
            raise FileNotFoundError(data / name)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"choose a new or empty experiment directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    common = [
        "--runtimes", str(data / "runtimes.csv"), "--token-ids", str(data / "token_ids.json"),
        "--device", args.device, "--batch-size", str(args.batch_size),
        "--latency-warmup", str(args.latency_warmup),
        "--latency-repetitions", str(args.latency_repetitions),
    ]
    rows, provenance = [], None
    conditions = [("baseline", 0.0), *[("adaptive", p) for p in args.length_penalties]]
    for seed in args.seeds:
        for architecture, penalty in conditions:
            tag = f"{architecture}_w{args.lstm_hidden_size}_lambda{penalty:g}_seed{seed}"
            checkpoint = output / f"{tag}.pt"
            command = [
                sys.executable, str(scripts / "train_adaptive.py"), *common,
                "--embedding-matrix", str(data / "fasttext_embeddings.pt"),
                "--architecture", architecture, "--lstm-hidden-size", str(args.lstm_hidden_size),
                "--length-penalty", str(penalty), "--problem-columns", "m", "n", "k",
                "--split-groups", str(data / "split_groups.json"),
                "--validation-groups", str(args.validation_groups), "--split-seed", str(args.split_seed),
                "--seed", str(seed), "--epochs", str(args.epochs), "--loss", args.loss,
                "--checkpoint", str(checkpoint), "--save-best", "--verbose",
            ]
            if args.dataset_run_dir:
                command.extend(("--dataset-run-dir", args.dataset_run_dir))
            if args.wandb_project:
                command.extend(("--wandb-project", args.wandb_project, "--wandb-run-name", tag,
                                "--wandb-mode", args.wandb_mode))
                if args.wandb_entity:
                    command.extend(("--wandb-entity", args.wandb_entity))
            print(f"[retention:run] {tag}", flush=True)
            subprocess.run(command, check=True)
            final = json.loads(checkpoint.with_suffix(".metrics.json").read_text())
            fields = ("split", "problem_columns", "split_groups", "action_catalog",
                      "input_fingerprints", "embedding_sha256", "lstm_hidden_size")
            current = {key: final[key] for key in fields}
            if provenance is not None and current != provenance:
                raise ValueError("data, width, or validation split changed between runs")
            provenance = current
            best_path = Path(final["best_checkpoint"])
            evaluated_path = output / f"{tag}.best.evaluation.json"
            subprocess.run([
                sys.executable, str(scripts / "evaluate_adaptive.py"), *common,
                "--checkpoint", str(best_path), "--output", str(evaluated_path),
            ], check=True)
            best = json.loads(evaluated_path.read_text())["model"]
            with checkpoint.with_suffix(".lengths.csv").open(newline="") as handle:
                length_rows = list(csv.DictReader(handle))
            best_persistence = persistent_length_evidence(length_rows, through_epoch=best["epoch"])
            best_signal = retention_signal(best["retention"], best_persistence)
            final_signal = retention_signal(final["retention"], final["retention_persistence"])
            rows.append({
                "architecture": architecture, "length_penalty": penalty, "seed": seed,
                "split_seed": args.split_seed, "best_epoch": best["epoch"],
                "best_checkpoint": str(best_path), "final_checkpoint": str(checkpoint),
                "best_evaluation": str(evaluated_path),
                "best_mean_regret": best["summary"]["mean_regret"],
                "final_mean_regret": final["validation"]["mean_regret"],
                "best_distinct_h": json.dumps(best["retention"]["lengths"]["distinct_h"])
                if best["retention"] else "[1]",
                "best_extra_state_regret_gain": best["retention"]["extra_state_regret_gain"]
                if best["retention"] else None,
                "best_variable_output_signal": best_signal["variable_output_signal"],
                "best_extra_states_help_signal": best_signal["extra_states_help_signal"],
                "final_variable_output_signal": final_signal["variable_output_signal"],
                "final_extra_states_help_signal": final_signal["extra_states_help_signal"],
                "best_retention": best["retention"], "best_persistence": best_persistence,
                "final_retention": final["retention"], "final_persistence": final["retention_persistence"],
            })
    baselines = {r["seed"]: r["best_mean_regret"] for r in rows if r["architecture"] == "baseline"}
    for row in rows:
        row["best_regret_minus_matched_baseline"] = row["best_mean_regret"] - baselines[row["seed"]]
        row["useful_variable_output_signal"] = (
            row["best_extra_states_help_signal"] and row["best_regret_minus_matched_baseline"] <= 1e-6
        )
    replicated = []
    for penalty in args.length_penalties:
        selected = [r for r in rows if r["architecture"] == "adaptive" and r["length_penalty"] == penalty]
        replicated.append({
            "length_penalty": penalty, "seed_count": len(selected),
            "positive_seed_count": sum(r["useful_variable_output_signal"] for r in selected),
            "replication_assessed": len(selected) >= 3,
            "replicated_useful_signal": all(r["useful_variable_output_signal"] for r in selected)
            if len(selected) >= 3 else None,
        })
    report = {
        "protocol": {**vars(args), "checkpoint_selection": "validation_regret_then_task_loss",
                     "persistence_window": 5, "minimum_T_or_diversity_reward": False,
                     "statistical_significance_established": False},
        "provenance": provenance, "runs": rows, "replication": replicated,
    }
    path = output / "experiment.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    columns = [key for key in rows[0] if not key.endswith(("_retention", "_persistence"))]
    with (output / "experiment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"report": str(path), "runs": len(rows), "replication": replicated}, indent=2))


if __name__ == "__main__":
    main()
