import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--cuda-dir",
    )
    inputs.add_argument(
        "--source-manifest",
    )
    parser.add_argument(
        "--source-root",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clang", default="clang++")
    parser.add_argument("--cuda-path")
    return parser.parse_args()


def main() -> None:
    from frontend.llvm_compiler import LLVMCompiler
    from frontend.tokenizer import LLVMTokenizer, Vocabulary

    args = parse_args()
    source_root = Path(
        args.source_root or args.cuda_dir or Path(args.source_manifest).parent
    ).resolve()
    output_root = Path(args.output_dir)
    ir_root = output_root / "llvm_ir"
    output_root.mkdir(parents=True, exist_ok=True)

    if args.source_manifest:
        manifest = json.loads(Path(args.source_manifest).read_text(encoding="utf-8"))
        source_items = []
        for kernel_id, relative in sorted(manifest.items()):
            source = (source_root / relative).resolve()
            source_items.append((kernel_id, source))
    else:
        sources = sorted(source_root.rglob("*.cu"))
        source_items = [
            (source.relative_to(source_root).with_suffix("").as_posix(), source)
            for source in sources
        ]
    
    print(f"we found {len(source_items)} cuda files, source {source_root}")

    compiler = LLVMCompiler(
        clang=args.clang,
        cuda_path=args.cuda_path,
        gpu_arch="sm_75",
    )
    tokenizer = LLVMTokenizer()
    documents = {}
    failures = {}
    for kernel_id, source in source_items:
        ir_path = ir_root / f"{kernel_id}.ll"
        try:
            compiler.compile(source, ir_path)
            documents[kernel_id] = tokenizer.tokenize_file(ir_path)
        except (OSError, RuntimeError) as exc:
            print(f"ERRRRORRR {kernel_id}: {exc}")
            failures[kernel_id] = str(exc)

    vocabulary = Vocabulary.build(
        documents.values(),
        min_frequency=1,
    )
    vocabulary.save(output_root / "vocabulary.json")
    print(f"vOcab length {len(vocabulary.token_to_id)}")
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
    
    print(
        len(documents), len(failures), len(vocabulary.token_to_id),
    )


if __name__ == "__main__":
    main()
