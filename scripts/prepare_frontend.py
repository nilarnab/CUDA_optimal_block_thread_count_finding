#!/usr/bin/env python3
"""Compile CUDA files to LLVM IR and create reusable token artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda-dir", required=True, help="directory searched recursively for .cu files"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clang", default="clang++")
    parser.add_argument("--cuda-path")
    parser.add_argument("--gpu-arch", default="sm_75")
    parser.add_argument("--min-frequency", type=int, default=1)
    parser.add_argument("--max-vocabulary", type=int)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="record compilation failures and continue processing other kernels",
    )
    return parser.parse_args()


def main() -> None:
    from frontend.llvm_compiler import LLVMCompiler
    from frontend.tokenizer import LLVMTokenizer, Vocabulary

    args = parse_args()
    source_root = Path(args.cuda_dir).resolve()
    output_root = Path(args.output_dir)
    ir_root = output_root / "llvm_ir"
    output_root.mkdir(parents=True, exist_ok=True)

    sources = sorted(source_root.rglob("*.cu"))
    if not sources:
        raise SystemExit(f"no .cu files found under {source_root}")
    if args.verbose:
        print(f"[frontend:discover] found {len(sources)} CUDA files under {source_root}")
        print(f"[frontend:config] GPU architecture: {args.gpu_arch}")
        print(f"[frontend:config] LLVM output directory: {ir_root}")

    compiler = LLVMCompiler(
        clang=args.clang,
        cuda_path=args.cuda_path,
        gpu_arch=args.gpu_arch,
        verbose=args.verbose,
    )
    tokenizer = LLVMTokenizer()
    documents: dict[str, list[str]] = {}
    failures: dict[str, str] = {}
    for source in sources:
        relative = source.relative_to(source_root)
        kernel_id = relative.with_suffix("").as_posix()
        ir_path = (ir_root / relative).with_suffix(".ll")
        try:
            if args.verbose:
                print(f"[frontend:compile] {kernel_id}: {source}")
            compiler.compile(source, ir_path)
            documents[kernel_id] = tokenizer.tokenize_file(ir_path)
            if args.verbose:
                print(
                    f"[frontend:tokenize] {kernel_id}: "
                    f"{len(documents[kernel_id])} tokens"
                )
        except (OSError, RuntimeError) as exc:
            if args.verbose:
                print(f"[frontend:failure] {kernel_id}: {exc}")
            if not args.keep_going:
                raise
            failures[kernel_id] = str(exc)

    if not documents:
        raise SystemExit("no CUDA source was successfully compiled")
    vocabulary = Vocabulary.build(
        documents.values(),
        min_frequency=args.min_frequency,
        max_size=args.max_vocabulary,
    )
    vocabulary.save(output_root / "vocabulary.json")
    if args.verbose:
        print(f"[frontend:vocabulary] {len(vocabulary.token_to_id)} entries")
    token_ids = {
        kernel_id: list(vocabulary.encode(tokens))
        for kernel_id, tokens in documents.items()
    }
    (output_root / "token_ids.json").write_text(
        json.dumps(token_ids, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (output_root / "fasttext_corpus.txt").open("w", encoding="utf-8") as corpus:
        for tokens in documents.values():
            corpus.write(" ".join(tokens) + "\n")
    (output_root / "compile_failures.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.verbose:
        print(f"[frontend:output] {output_root / 'vocabulary.json'}")
        print(f"[frontend:output] {output_root / 'token_ids.json'}")
        print(f"[frontend:output] {output_root / 'fasttext_corpus.txt'}")
    print(
        json.dumps(
            {
                "compiled": len(documents),
                "failed": len(failures),
                "vocabulary_size": len(vocabulary.token_to_id),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
