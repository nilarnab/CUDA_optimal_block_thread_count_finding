#!/usr/bin/env python3
"""Load a trained selector, choose blocks for vector addition, and verify CUDA output."""

import argparse
import ctypes as C
from pathlib import Path

import torch

from inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--vocabulary", required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--side", type=int, nargs="+", required=True, help="M values for M*M elements")
    parser.add_argument("--gpu", default="cuda:0")
    parser.add_argument("--predictor-device", default="cpu")
    args = parser.parse_args()
    if min(args.side) < 1:
        parser.error("side lengths must be positive")
    gpu = torch.device(args.gpu)
    if gpu.type != "cuda" or not torch.cuda.is_available():
        parser.error("this example requires an available NVIDIA CUDA device")

    engine = InferenceEngine.from_checkpoint(
        args.checkpoint, args.vocabulary, device=args.predictor_device
    )
    if engine.problem_columns != ("m", "n", "k"):
        parser.error("this example expects a checkpoint trained on M/N/K context")
    cached = engine.prepare_kernel((args.build_dir / "kernel.ll").read_text(encoding="utf-8"))
    library = C.CDLL(str((args.build_dir / "kernel.so").resolve()))
    library.launch_limits.argtypes = [C.POINTER(C.c_int), C.POINTER(C.c_int)]
    library.launch_limits.restype = C.c_int
    library.launch_vector_add.argtypes = [
        C.c_void_p, C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_void_p,
    ]
    library.launch_vector_add.restype = C.c_int
    library.launch_error.argtypes = [C.c_int]
    library.launch_error.restype = C.c_char_p

    def check(status):
        if status:
            raise RuntimeError(library.launch_error(status).decode())

    with torch.cuda.device(gpu):
        max_threads, max_grid = C.c_int(), C.c_int()
        check(library.launch_limits(C.byref(max_threads), C.byref(max_grid)))
        stream = torch.cuda.current_stream(gpu)
        for side in args.side:
            n = side * side
            allowed = [config for config in engine.catalog
                       if config.y == config.z == 1 and config.x <= max_threads.value
                       and 1 + (n - 1) // config.x <= max_grid.value]
            prediction = engine.predict_cached(cached, (side, 1, 1), allowed=allowed)
            threads = prediction.configuration.x
            a = torch.ones(n, device=gpu, dtype=torch.float32)
            b = torch.full_like(a, 2)
            out = torch.empty_like(a)
            check(library.launch_vector_add(
                a.data_ptr(), b.data_ptr(), out.data_ptr(), n, threads, stream.cuda_stream
            ))
            stream.synchronize()
            torch.testing.assert_close(out, torch.full_like(out, 3), rtol=0, atol=0)
            print(f"M={side} elements={n} block_x={threads} grid_x={1 + (n - 1) // threads} verified")


if __name__ == "__main__":
    main()
