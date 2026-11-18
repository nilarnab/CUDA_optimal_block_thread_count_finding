#!/usr/bin/env python3
"""Import the official LS-CAT T4 archive as a complete 16-action dataset."""

from __future__ import annotations

import argparse
import json


def main() -> None:
    from data.lscat_adapter import import_lscat_1d

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="data/raw/ls-cat-official")
    parser.add_argument("--archive")
    parser.add_argument("--output-dir", default="data/processed/lscat_t4_1d")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    summary = import_lscat_1d(
        args.repository,
        args.output_dir,
        archive=args.archive,
    )
    if args.verbose:
        print(f"[lscat:output] {args.output_dir}/runtimes.csv")
        print(f"[lscat:output] {args.output_dir}/split_groups.json")
        print(f"[lscat:output] {args.output_dir}/sources.json")
        print("[lscat:frontend] CUDA sources remain in the official repository")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
