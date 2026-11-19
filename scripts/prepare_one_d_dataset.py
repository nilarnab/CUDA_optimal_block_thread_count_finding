#!/usr/bin/env python3
"""Convert a twelve-kernel 1D run into M/N/K runtime targets and source inputs."""

import argparse
import json


def main():
    from data.one_d_adapter import prepare_one_d_dataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed_1d")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    summary = prepare_one_d_dataset(args.run_dir, args.output_dir)
    if args.verbose:
        print(f"[one-d:output] {args.output_dir}/runtimes.csv (M/N/K, microseconds)")
        print(f"[one-d:output] {args.output_dir}/cuda (one source per base kernel)")
        print(f"[one-d:output] {args.output_dir}/split_groups.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
