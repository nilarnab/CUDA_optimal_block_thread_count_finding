#!/usr/bin/env python3
"""Prepare the collected LS-CAT pilot for the 16-action NLP baseline."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    from data.pilot_adapter import prepare_pilot_dataset

    args = parse_args()
    if args.verbose:
        print(f"[pilot:input] run directory: {args.run_dir}")
        print(f"[pilot:output] processed directory: {args.output_dir}")
        print("[pilot:actions] retaining 16 scalar actions from 64 through 1024")
        print("[pilot:actions] mapping native (N,1) and (1,N) to abstract action N")
    summary = prepare_pilot_dataset(args.run_dir, args.output_dir)
    if args.verbose:
        print(f"[pilot:output] runtime table: {args.output_dir}/runtimes.csv")
        print(f"[pilot:output] CUDA sources: {args.output_dir}/cuda")
        print(f"[pilot:output] launch mappings: {args.output_dir}/launch_mappings.json")
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
