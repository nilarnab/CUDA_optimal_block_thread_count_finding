#!/usr/bin/env python3
"""Summarize thermal launch-configuration experiment results."""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, help="CSV emitted by the experiment")
    parser.add_argument(
        "--summary", type=Path, help="optionally write the aggregate table as CSV"
    )
    args = parser.parse_args()

    grouped: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    with args.results.open(newline="") as source:
        for row in csv.DictReader(source):
            key = (
                int(row["target_temp_c"]),
                int(row["threads"]),
                int(row["blocks_per_sm"]),
            )
            grouped[key].append(float(row["runtime_ms"]))

    if not grouped:
        raise SystemExit("No trial rows found")

    aggregate = []
    targets = sorted({key[0] for key in grouped})
    for target in targets:
        rows = []
        for (row_target, threads, blocks_per_sm), samples in grouped.items():
            if row_target != target:
                continue
            median_ms = statistics.median(samples)
            rows.append(
                {
                    "target_temp_c": target,
                    "threads": threads,
                    "blocks_per_sm": blocks_per_sm,
                    "samples": len(samples),
                    "median_ms": median_ms,
                    "min_ms": min(samples),
                    "max_ms": max(samples),
                    "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
                }
            )

        best_ms = min(row["median_ms"] for row in rows)
        for row in rows:
            row["regret_pct"] = (row["median_ms"] / best_ms - 1.0) * 100.0
        rows.sort(key=lambda row: row["median_ms"])
        aggregate.extend(rows)

        print(f"\nTarget {target} C")
        print("threads  blocks/SM  n  median_ms  stdev_ms  regret")
        for index, row in enumerate(rows):
            marker = "  <-- best" if index == 0 else ""
            print(
                f"{row['threads']:7d}  {row['blocks_per_sm']:9d}  "
                f"{row['samples']:2d}  {row['median_ms']:9.3f}  "
                f"{row['stdev_ms']:8.3f}  {row['regret_pct']:6.2f}%{marker}"
            )

    best_by_target = {
        target: min(
            (row for row in aggregate if row["target_temp_c"] == target),
            key=lambda row: row["median_ms"],
        )
        for target in targets
    }
    best_configs = {
        (row["threads"], row["blocks_per_sm"])
        for row in best_by_target.values()
    }
    print("\nBest-configuration result:")
    if len(best_configs) == 1:
        threads, blocks_per_sm = next(iter(best_configs))
        print(
            "  No ranking change observed: "
            f"{threads} threads and {blocks_per_sm} blocks/SM was best at every target."
        )
    else:
        print("  The point-estimate winner changed across target temperatures:")
        for target in targets:
            row = best_by_target[target]
            print(
                f"    {target} C: {row['threads']} threads, "
                f"{row['blocks_per_sm']} blocks/SM ({row['median_ms']:.3f} ms)"
            )
        print("  Check variability before treating this as a statistically real change.")

    if args.summary:
        fieldnames = [
            "target_temp_c",
            "threads",
            "blocks_per_sm",
            "samples",
            "median_ms",
            "stdev_ms",
            "min_ms",
            "max_ms",
            "regret_pct",
        ]
        with args.summary.open("w", newline="") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregate)
        print(f"\nWrote {args.summary}")


if __name__ == "__main__":
    main()
