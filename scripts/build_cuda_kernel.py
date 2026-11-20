#!/usr/bin/env python3
"""Build textual LLVM IR and a Linux CUDA launcher library for one kernel."""

import argparse
from pathlib import Path
import subprocess
import tempfile

from frontend.llvm_compiler import LLVMCompiler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", type=Path, required=True, help="kernel-only CUDA source")
    parser.add_argument("--launcher", type=Path, required=True, help="CUDA host bridge that includes the kernel")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory for kernel.ll and kernel.so")
    parser.add_argument("--gpu-arch", required=True, help="CUDA target such as sm_75")
    parser.add_argument("--clang", default="clang++")
    parser.add_argument("--nvcc", default="nvcc")
    parser.add_argument("--cuda-path")
    args = parser.parse_args()

    for source in (args.kernel, args.launcher):
        if not source.is_file():
            parser.error(f"source file does not exist: {source}")
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    compiler = LLVMCompiler(clang=args.clang, cuda_path=args.cuda_path, gpu_arch=args.gpu_arch)
    with tempfile.TemporaryDirectory(dir=args.output_dir.parent) as temp:
        build = Path(temp) / "build"
        compiler.compile(args.kernel.resolve(), build / "kernel.ll")
        subprocess.run([
            args.nvcc, "-O3", f"-arch={args.gpu_arch}", "--shared", "-Xcompiler=-fPIC",
            str(args.launcher.resolve()), "-o", str(build / "kernel.so"),
        ], check=True)
        build.rename(args.output_dir)
    print(f"Built {args.output_dir / 'kernel.ll'} and {args.output_dir / 'kernel.so'}")


if __name__ == "__main__":
    main()
