import argparse
from dataclasses import asdict
import json
from pathlib import Path

from train_adaptive import build_model, load_datasets, make_loader


def evaluate_checkpoint(path, args):
    import torch
    from training.adaptive_evaluation import case_summary, length_summary, prediction_cases, timing_environment
    from training.retention_evaluation import retention_diagnostics

    saved = torch.load(path, map_location="cpu", weights_only=True)
    metadata = saved["metadata"]

    metadata.setdefault("problem_columns", ["problem_size"])
    metadata.setdefault("split_groups", None)

    matrix = saved["model_state"]["token_encoder.embedding.weight"]
    catalog, split, _, validation = load_datasets(
        args.runtimes, args.token_ids, matrix, metadata["target_exponent"], split=metadata["split"],
        problem_columns=metadata["problem_columns"], split_groups=metadata["split_groups"],
    )

    model = build_model(matrix, metadata["context_size"], len(catalog), metadata["architecture"],
                        lstm_hidden_size=metadata.get("lstm_hidden_size", 128))
    model.load_state_dict(saved["model_state"])
    model.to(args.device).eval()
    cases = prediction_cases(
        model, make_loader(validation, args.batch_size), catalog, args.device,
        loss_kind=saved["trainer_config"]["loss"], length_penalty=metadata.get("length_penalty", 0.0),
        warmup=args.latency_warmup, repetitions=args.latency_repetitions,
        epoch=metadata.get("history", [{"epoch": 0}])[-1]["epoch"],
    )
    retention = retention_diagnostics(
        model, make_loader(validation, args.batch_size), catalog, args.device,
        loss_kind=saved["trainer_config"]["loss"], epoch=metadata["history"][-1]["epoch"],
    )
    return {
        "checkpoint": str(path), "architecture": metadata["architecture"],
        "lstm_hidden_size": metadata.get("lstm_hidden_size", 128),
        "epoch": metadata["history"][-1]["epoch"],
        "checkpoint_selection": metadata.get("checkpoint_selection", "final"),
        "loss_kind": saved["trainer_config"]["loss"], "length_penalty": metadata.get("length_penalty", 0.0),
        "validation_kernels": list(split["validation"]), "problem_columns": metadata["problem_columns"],
        "timing_environment": timing_environment(args.device, metadata.get("dataset_device")),
        "summary": case_summary(cases), "compression": length_summary(cases), "cases": cases,
        "retention": retention,
    }, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--compare")
    parser.add_argument("--runtimes", required=True)
    parser.add_argument("--token-ids", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--latency-warmup", type=int, default=3)
    parser.add_argument("--latency-repetitions", type=int, default=10)
    parser.add_argument("--output", default="checkpoints/adaptive_comparison.json")
    args = parser.parse_args()

    from training.adaptive_evaluation import write_rows

    result, metadata = evaluate_checkpoint(args.checkpoint, args)
    report = {"model": result}
    if args.compare:
        comparison, baseline = evaluate_checkpoint(args.compare, args)
        report["comparison"] = comparison
        report["delta_model_minus_comparison"] = {
            name: result["summary"][name] - comparison["summary"][name]
            for name in ("performance_score", "mean_regret", "exact_optimal_rate")
        }
        report["speedup_comparison_over_model"] = {
            name: sum(row[name] for row in comparison["cases"]) / sum(row[name] for row in result["cases"])
            for name in ("selected_kernel_ms", "dense_plus_kernel_ms", "estimated_total_ms")
        }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_rows(path.with_suffix(".model.csv"), result["cases"], result["problem_columns"])
    if args.compare:
        write_rows(path.with_suffix(".comparison.csv"), comparison["cases"], comparison["problem_columns"])
    print({"report": str(path), **report})


if __name__ == "__main__":
    main()
