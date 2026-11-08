"""Compile CUDA source into textual LLVM IR with Clang."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class LLVMCompiler:
    clang: str = "clang++"
    cuda_path: str | None = None
    gpu_arch: str = "sm_75"
    optimization: str = "-O3"
    extra_args: Sequence[str] = field(default_factory=tuple)
    verbose: bool = False

    def compile(self, source: str | Path, output: str | Path) -> Path:
        """Emit device-side LLVM IR for one CUDA translation unit."""

        source_path = Path(source)
        output_path = Path(output)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.clang,
            "-x",
            "cuda",
            "--cuda-device-only",
            f"--cuda-gpu-arch={self.gpu_arch}",
            "-S",
            "-emit-llvm",
            self.optimization,
        ]
        if self.cuda_path:
            command.append(f"--cuda-path={self.cuda_path}")
        command.extend(self.extra_args)
        command.extend([str(source_path), "-o", str(output_path)])
        if self.verbose:
            print(f"[frontend:clang] {' '.join(command)}")
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if self.verbose and completed.stdout.strip():
            print(f"[frontend:clang:stdout]\n{completed.stdout.strip()}")
        if self.verbose and completed.stderr.strip():
            print(f"[frontend:clang:stderr]\n{completed.stderr.strip()}")
        if completed.returncode != 0:
            raise RuntimeError(
                f"LLVM compilation failed for {source_path}:\n{completed.stderr.strip()}"
            )
        return output_path

