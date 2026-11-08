#!/usr/bin/env python3
"""Train the frozen skip-gram frontend and export its PyTorch matrix."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--output-model", required=True)
    parser.add_argument("--output-matrix", required=True)
    parser.add_argument("--dimensions", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--minimum-count", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import fasttext
        import torch
    except ImportError as exc:
        raise SystemExit("install the fasttext optional dependencies first") from exc

    from encoders.fasttext_encoder import embedding_matrix_from_fasttext
    from frontend.tokenizer import Vocabulary

    if args.verbose:
        print(f"[fasttext:input] corpus: {args.corpus}")
        print(f"[fasttext:input] vocabulary: {args.vocabulary}")
        print(
            f"[fasttext:config] skipgram dimensions={args.dimensions} "
            f"epochs={args.epochs} threads={args.threads}"
        )

    model = fasttext.train_unsupervised(
        args.corpus,
        model="skipgram",
        dim=args.dimensions,
        epoch=args.epochs,
        minCount=args.minimum_count,
        thread=args.threads,
        verbose=2 if args.verbose else 0,
    )
    model_path = Path(args.output_model)
    matrix_path = Path(args.output_matrix)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    vocabulary = Vocabulary.load(args.vocabulary)
    if args.verbose:
        print(f"[fasttext:matrix] exporting {len(vocabulary.token_to_id)} token vectors")
    matrix = embedding_matrix_from_fasttext(model_path, vocabulary.token_to_id)
    torch.save(matrix, matrix_path)
    print(f"saved FastText model to {model_path}")
    print(f"saved {tuple(matrix.shape)} embedding matrix to {matrix_path}")


if __name__ == "__main__":
    main()
