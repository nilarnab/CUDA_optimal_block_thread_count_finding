#!/usr/bin/env python3
"""Recover frontend artifacts from existing LLVM IR and filter failed kernels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    from frontend.tokenizer import LLVMTokenizer, Vocabulary

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ir-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runtimes", required=True)
    parser.add_argument("--split-groups", required=True)
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--max-vocabulary", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    ir_root = Path(args.ir_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    ir_files = sorted(ir_root.rglob("*.ll"))
    if not ir_files:
        raise SystemExit(f"no LLVM IR files found under {ir_root}")

    tokenizer = LLVMTokenizer()
    documents = {}
    for index, path in enumerate(ir_files, start=1):
        kernel_id = path.relative_to(ir_root).with_suffix("").as_posix()
        documents[kernel_id] = tokenizer.tokenize_file(path)
        if args.verbose and (index == 1 or index % 500 == 0 or index == len(ir_files)):
            print(f"[ir-frontend:tokenize] {index}/{len(ir_files)} {kernel_id}")

    vocabulary = Vocabulary.build(
        documents.values(),
        min_frequency=args.min_frequency,
        max_size=args.max_vocabulary,
    )
    vocabulary.save(output / "vocabulary.json")
    token_ids = {
        kernel_id: list(vocabulary.encode(tokens))
        for kernel_id, tokens in documents.items()
    }
    (output / "token_ids.json").write_text(
        json.dumps(token_ids, separators=(",", ":")), encoding="utf-8"
    )
    with (output / "fasttext_corpus.txt").open("w", encoding="utf-8") as handle:
        for tokens in documents.values():
            handle.write(" ".join(tokens) + "\n")

    groups = json.loads(Path(args.split_groups).read_text(encoding="utf-8"))
    if not isinstance(groups, dict):
        raise ValueError("split groups must be a JSON object")
    retained = set(token_ids)
    filtered_groups = {key: value for key, value in groups.items() if key in retained}
    if set(filtered_groups) != retained:
        missing = sorted(retained.difference(filtered_groups))
        raise ValueError(f"LLVM kernels missing from split groups: {missing[:5]}")
    group_output = output / "split_groups_frontend.json"
    group_output.write_text(
        json.dumps(filtered_groups, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    runtime_output = output / "runtimes_frontend.csv"
    runtime_rows = contexts = 0
    previous_context = None
    with Path(args.runtimes).open(newline="", encoding="utf-8") as source, \
            runtime_output.open("w", newline="", encoding="utf-8") as destination:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or "kernel_id" not in reader.fieldnames:
            raise ValueError("runtime CSV must contain kernel_id")
        writer = csv.DictWriter(destination, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["kernel_id"] not in retained:
                continue
            writer.writerow(row)
            runtime_rows += 1
            context = (row["kernel_id"], row.get("m"), row.get("n"))
            if context != previous_context:
                contexts += 1
                previous_context = context

    summary = {
        "llvm_kernels": len(ir_files),
        "vocabulary_size": len(vocabulary.token_to_id),
        "runtime_rows": runtime_rows,
        "contexts": contexts,
        "source_groups": len(set(filtered_groups.values())),
        "token_ids": str(output / "token_ids.json"),
        "fasttext_corpus": str(output / "fasttext_corpus.txt"),
        "runtimes": str(runtime_output),
        "split_groups": str(group_output),
    }
    (output / "frontend_dataset.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
