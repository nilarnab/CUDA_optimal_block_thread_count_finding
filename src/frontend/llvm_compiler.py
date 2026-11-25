from __future__ import annotations

import os
import shutil
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

    def _cuda_root(self):
        if self.cuda_path:
            return Path(self.cuda_path)
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix and (Path(conda_prefix) / "bin" / "nvcc").is_file():
            return Path(conda_prefix)
        nvcc = shutil.which("nvcc")
        if nvcc:
            return Path(nvcc).resolve().parent.parent
        return None

    @staticmethod
    def _target_include(cuda_root):
        candidates = sorted((cuda_root / "targets").glob("*/include"))
        return next((path for path in candidates if (path / "cuda.h").is_file()), None)

    def compile(self, source, output):
        source_path = Path(source)
        output_path = Path(output)
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
        cuda_root = self._cuda_root()
        if cuda_root:
            command.append(f"--cuda-path={cuda_root}")
            target_include = self._target_include(cuda_root)
            if target_include:
                command.extend(("-I", str(target_include)))
        command.extend(self.extra_args)
        command.extend([str(source_path), "-o", str(output_path)])
        if self.verbose:
            print(f"[frontend:clang] {' '.join(command)}")
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if self.verbose and completed.stdout.strip():
            print(f"[frontend:clang:stdout]\n{completed.stdout.strip()}")
        if self.verbose and completed.stderr.strip():
            print(f"[frontend:clang:stderr]\n{completed.stderr.strip()}")
        return output_path
