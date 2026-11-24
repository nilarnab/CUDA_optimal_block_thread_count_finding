import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shlex
from statistics import median
import subprocess
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from frontend.tokenizer import LLVMTokenizer
from launch.configuration_catalog import ConfigurationCatalog


DEFAULT_SIZES = [(150051, 1500), (180051, 1500), (150051, 2000), (200051, 2000)]
DEFAULT_BLOCKS = [config.x for config in ConfigurationCatalog.lscat()]
WARP_THREADS = 32
NEARBY_BLOCK_RADIUS = 128
METRICS = {
    "total_us": "Total launch latency (µs)",
    "gpu_us": "GPU kernel time (µs)",
    "selection_us": "CPU selection time (µs)",
}
METADATA_FIELDS = ("retained_states", "full_states", "state_reduction_pct")
FIELDS = (
    "kernel", "problem_size", "mode", "repeat", "block_size", "grid", "trials",
    *METRICS, "process_wall_s", *METADATA_FIELDS,
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def fixed_sweep(predicted, blocks):
    candidates = set(DEFAULT_BLOCKS if blocks is None else blocks)
    if blocks is None:
        first = max(WARP_THREADS, predicted - NEARBY_BLOCK_RADIUS)
        last = min(DEFAULT_BLOCKS[-1], predicted + NEARBY_BLOCK_RADIUS)
        candidates.update(range(first, last + 1, WARP_THREADS))
    return sorted(candidates | {predicted})


def state_counts(binary, manifest):
    if manifest is None:
        # The executable links to exactly one toolcall build via its ELF runpath.
        try:
            dynamic = subprocess.run(
                ["readelf", "-dW", str(binary)], capture_output=True, text=True, check=True
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            dynamic = ""
        match = re.search(r"\$ORIGIN/(\.toolcall/kernel-[\w-]+)", dynamic)
        if match:
            manifest = binary.parent / match[1] / "build.json"
        if manifest is None or not manifest.is_file():
            print("No toolcall build.json found. Pass --build-json to record gate counts.", file=sys.stderr)
            return {}

    retained = json.loads(manifest.read_text(encoding="utf-8"))["retained_states"]
    if type(retained) is not int or retained < 1:
        raise ValueError(f"invalid retained_states in {manifest}")
    ir = manifest.with_name("kernel.ll")
    if not ir.is_file():
        print(f"Missing {ir}. Recording retained states only.", file=sys.stderr)
        return {"retained_states": retained}
    # Inference uses one state per token and substitutes a single UNK for empty IR.
    full = max(1, len(LLVMTokenizer().tokenize_file(ir)))
    if retained > full:
        raise ValueError(f"{manifest} reports {retained} retained states but {ir} has only {full}")
    reduction = round(100 * (full - retained) / full, 3)
    print(f"Kernel states: {retained}/{full} retained ({reduction:g}% fewer rows)", flush=True)
    return dict(zip(METADATA_FIELDS, (retained, full, reduction)))


def measure(binary, size, fixed_block=None):
    mode = "model" if fixed_block is None else "fixed"
    command = [str(binary), *map(str, size)]
    if fixed_block is not None:
        command.append(str(fixed_block))
    start = perf_counter()
    result = subprocess.run(command, capture_output=True, text=True)
    elapsed = perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"{shlex.join(command)} exited {result.returncode}\n{result.stdout}{result.stderr}")

    lines = result.stdout.splitlines()
    header = next((line for line in lines if line.startswith(mode + " ") and line.endswith(" verified")), "")
    timing = next((line for line in lines if line.startswith("gpu_us=")), "")
    fields = dict(re.findall(r"(\w+)=([^\s]+)", header + " " + timing))
    try:
        block, grid, trials = (int(fields[key]) for key in ("block", "grid", "trials"))
        times = {key: float(fields[key]) for key in METRICS}
        if min(block, grid, trials) < 1 or any(not math.isfinite(t) or t < 0 for t in times.values()):
            raise ValueError("invalid benchmark measurements")
        if fixed_block is not None and block != fixed_block:
            raise ValueError(f"expected fixed block {fixed_block}, got {block}")
    except (KeyError, ValueError) as error:
        raise RuntimeError(f"Unexpected benchmark output from {shlex.join(command)}\n{result.stdout}{result.stderr}") from error
    return {"mode": mode, "block_size": block, "grid": grid, "trials": trials,
            **times, "process_wall_s": round(elapsed, 6)}


def curves(groups, metric):
    for size, modes in groups.items():
        yield size, *(
            [(block, median(row[metric] for row in rows)) for block, rows in sorted(modes[mode].items())]
            for mode in ("fixed", "model")
        )


def save_plot(plt, path, groups):
    fig, axes = plt.subplots(len(METRICS), 1, figsize=(10, 10), sharex=True, layout="constrained")
    for axis, (metric, label) in zip(axes, METRICS.items()):
        for size, fixed, model in curves(groups, metric):
            (line,) = axis.plot(*zip(*fixed), marker=".", label=size)
            axis.scatter(*zip(*model), color=line.get_color(), marker="D", zorder=3)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
    axes[0].set_title("Fixed block sweep · Diamonds show measured model choices")
    axes[0].legend(title="Problem size")
    axes[-1].set_xlabel("CUDA threads per block")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as figure:
        fig.savefig(figure, format="png", dpi=150)
    plt.close(fig)
    print(f"Plot {path}")


def log_wandb(wandb, args, rows, groups, metadata):
    config = {
        "kernel": args.kernel, "problem_sizes": args.size, "fixed_blocks": args.blocks or DEFAULT_BLOCKS,
        "tested_blocks_by_size": {size: sorted(modes["fixed"]) for size, modes in groups.items()},
        "repeats": args.repeats, "model_binary": str(args.model_bin), "fixed_binary": str(args.fixed_bin),
        **metadata,
    }
    with wandb.init(project=args.wandb_project, entity=args.wandb_entity,
                    name=args.csv.stem, job_type="launch-benchmark", config=config) as run:
        artifact = wandb.Artifact(f"{args.kernel}-launch-benchmarks", type="benchmark")
        artifact.add_file(str(args.csv))
        run.log_artifact(artifact)
        plots = {"measurements": wandb.Table(
            columns=list(FIELDS), data=[[row.get(key) for key in FIELDS] for row in rows]
        )}
        for metric, label in METRICS.items():
            xs, ys, labels = [], [], []
            for size, fixed, model in curves(groups, metric):
                blocks, values = zip(*fixed)
                xs.append(list(blocks))
                ys.append(list(values))
                labels.append(f"{size} fixed")
                for block, value in model:
                    xs.append([blocks[0], blocks[-1]] if len(blocks) > 1 else [block])
                    ys.append([value] * len(xs[-1]))
                    labels.append(f"{size} model reference (block {block})")
            plots[metric] = wandb.plot.line_series(
                xs=xs, ys=ys, keys=labels, title=label, xname="CUDA threads per block"
            )
        run.log(plots)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--kernel", default="softmax", help="kernel label")
    parser.add_argument("--model-bin", type=Path, help="defaults to build/<kernel>_aot")
    parser.add_argument("--fixed-bin", type=Path, help="defaults to build/<kernel>_fixed")
    parser.add_argument("--size", metavar="DIM", type=positive_int, nargs="+", action="append",
                        help="positional size arguments to both binaries, repeat for each shape")
    parser.add_argument("--blocks", type=positive_int, nargs="+",
                        help="override the catalog sweep and its fine steps around the model choice")
    parser.add_argument("--repeats", type=positive_int, default=1, help="process runs per size and block")
    parser.add_argument("--build-json", type=Path, help="model binary's toolcall manifest (auto-detected on ELF)")
    parser.add_argument("--csv", type=Path, help="CSV path (defaults to a timestamped file under results/launch_benchmarks)")
    parser.add_argument("--plot", type=Path, metavar="PNG", help="also save a plot to this path")
    parser.add_argument("--wandb", action="store_true", help="log the CSV and plots to Weights & Biases")
    parser.add_argument("--wandb-project", default="gpus-project")
    parser.add_argument("--wandb-entity", default="sm12779-new-york-university")
    args = parser.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.kernel):
        parser.error("--kernel must use letters, digits, underscores or hyphens")
    if args.size is None and args.kernel != "softmax":
        parser.error("--size is required for kernels other than softmax")
    args.size = list(dict.fromkeys(tuple(size) for size in (args.size or DEFAULT_SIZES)))
    args.model_bin = (args.model_bin or Path("build") / f"{args.kernel}_aot").resolve()
    args.fixed_bin = (args.fixed_bin or Path("build") / f"{args.kernel}_fixed").resolve()
    args.csv = args.csv or (
        Path("results/launch_benchmarks") / f"{args.kernel}_{datetime.now():%Y%m%dT%H%M%S_%f}.csv"
    )
    for binary in (args.model_bin, args.fixed_bin):
        if not binary.is_file() or not os.access(binary, os.X_OK):
            parser.error(f"expected an executable benchmark binary at {binary}")
    if args.plot and (args.plot.suffix.lower() != ".png" or args.plot.exists()
                      or args.plot.resolve() == args.csv.resolve()):
        parser.error("--plot must be a new .png path distinct from the CSV")

    try:
        if args.wandb:
            import wandb
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

        metadata = state_counts(args.model_bin, args.build_json)
        groups = defaultdict(lambda: {"fixed": defaultdict(list), "model": defaultdict(list)})
        rows = []
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("x", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            output.flush()
            print(f"Writing measurements to {args.csv}", flush=True)

            def record(problem, repeat, result):
                row = {"kernel": args.kernel, "problem_size": problem, "repeat": repeat, **result, **metadata}
                writer.writerow(row)
                output.flush()
                rows.append(row)
                groups[problem][row["mode"]][row["block_size"]].append(row)

            for size in args.size:
                problem = "x".join(map(str, size))
                model_totals = []
                for repeat in range(1, args.repeats + 1):
                    print(f"{problem} repeat {repeat}/{args.repeats}: running model", flush=True)
                    model = measure(args.model_bin, size)
                    record(problem, repeat, model)
                    block = model["block_size"]
                    model_totals.append(model["total_us"])
                    print(f"  model picked {block}, total {model['total_us']:g} µs", flush=True)
                    for fixed in fixed_sweep(block, args.blocks):
                        print(f"  fixed block {fixed}", flush=True)
                        record(problem, repeat, measure(args.fixed_bin, size, fixed))
                if args.repeats > 1:
                    print(f"{problem} model total range {min(model_totals):g}–{max(model_totals):g} µs", flush=True)

        if args.plot:
            save_plot(plt, args.plot, groups)
        if args.wandb:
            try:
                log_wandb(wandb, args, rows, groups, metadata)
            except wandb.Error as error:
                parser.exit(1, f"W&B logging failed: {error}\nCSV saved to {args.csv}\n")
    except ImportError as error:
        parser.exit(1, f"{error}. Install wandb for --wandb or matplotlib for --plot.\n")
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
