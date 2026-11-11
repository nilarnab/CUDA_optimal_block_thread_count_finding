#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo 'Usage: bash run.sh [smoke|quick|full|seconds|calibrate] [CUDA-logical-device-index] [collector options]'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  usage
  exit 0
fi
mode="${1:-smoke}"
device="${2:-0}"
if [[ "$mode" != smoke && "$mode" != quick && "$mode" != calibrate && "$mode" != full && "$mode" != seconds ]] || [[ ! "$device" =~ ^[0-9]+$ ]]; then
  usage >&2
  exit 2
fi
if [[ "$(uname -s)" != Linux ]]; then
  echo 'Run this package on a Linux GPU compute host.' >&2
  exit 2
fi
host_name="$(hostname)"
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  case "$host_name" in
    access|access[0-9]*|access.cims.nyu.edu|access[0-9]*.cims.nyu.edu|log-[0-9]*|torch-login-*|greene-login*|login*.hpc.nyu.edu|greene.hpc.nyu.edu|torch.hpc.nyu.edu)
      echo 'This is an NYU access/login host. Enter an allocated GPU session or a CIMS CUDA compute host first.' >&2
      exit 2
      ;;
  esac
fi

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$script_dir/runs"
run_dir="$(mktemp -d "$script_dir/runs/heavy-${mode}-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"

gpu_inventory() {
  if command -v nvidia-smi >/dev/null; then
    if command -v timeout >/dev/null; then
      timeout 10s nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version --format=csv
    else
      nvidia-smi --query-gpu=index,uuid,pci.bus_id,name,driver_version --format=csv
    fi
  else
    echo 'nvidia-smi is unavailable.'
  fi
}
finish() {
  run_status=$?
  trap - EXIT
  set +e
  gpu_inventory > "$run_dir/nvidia-smi-after.txt" 2>&1
  {
    printf '\nFinished UTC: '
    date -u +%Y-%m-%dT%H:%M:%SZ
    printf 'Exit status: %s\n' "$run_status"
    cat "$run_dir/nvidia-smi-after.txt"
  } >> "$run_dir/environment.txt" 2>&1
  printf '%s\n' "$run_status" > "$run_dir/exit_status.txt"
  archive_path="${run_dir}.tar.gz"
  tar -czf "$archive_path" -C "$(dirname -- "$run_dir")" "$(basename -- "$run_dir")"
  archive_status=$?
  echo "Results: $run_dir"
  if (( archive_status == 0 )); then
    echo "Results archive: $archive_path"
  else
    echo 'Archiving failed; the complete or partial run directory is retained.' >&2
    if (( run_status == 0 )); then run_status=$archive_status; fi
  fi
  exit "$run_status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

{
  printf 'Started UTC: '
  date -u +%Y-%m-%dT%H:%M:%SZ
  printf 'Mode: %s\nCUDA logical device: %s\n' "$mode" "$device"
  hostname
  uname -srm
  command -v nvcc || true
  nvcc --version || true
  python3 --version || true
  cc --version || true
  c++ --version || true
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES-unset}"
  printf 'NVIDIA_VISIBLE_DEVICES=%s\n' "${NVIDIA_VISIBLE_DEVICES-unset}"
  printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID-unset}"
  printf 'SLURM_JOB_NODELIST=%s\n' "${SLURM_JOB_NODELIST-unset}"
  printf 'SLURM_JOB_GPUS=%s\n' "${SLURM_JOB_GPUS-unset}"
  printf 'SLURM_STEP_GPUS=%s\n' "${SLURM_STEP_GPUS-unset}"
  printf 'SLURM_CPUS_PER_TASK=%s\n' "${SLURM_CPUS_PER_TASK-unset}"
  printf 'SLURM_ARRAY_TASK_ID=%s\n' "${SLURM_ARRAY_TASK_ID-unset}"
} > "$run_dir/environment.txt" 2>&1
gpu_inventory > "$run_dir/nvidia-smi-before.txt" 2>&1 || true
cat "$run_dir/nvidia-smi-before.txt" >> "$run_dir/environment.txt"
echo "Results: $run_dir"
echo "CUDA_VISIBLE_DEVICES is preserved. Selected CUDA logical device: $device"
for required_command in python3 nvcc c++ tar tee; do
  if ! command -v "$required_command" >/dev/null; then
    echo "Missing $required_command. Load the available CUDA/compiler/Python environment, then retry. See $run_dir/environment.txt" >&2
    exit 1
  fi
done
if ! python3 - > "$run_dir/python_preflight.log" 2>&1 <<'PY'
import sys
if sys.version_info < (3, 9):
    raise SystemExit("Python 3.9 or newer is required; found " + sys.version.split()[0])
print("Python version accepted:", sys.version.split()[0])
PY
then
  cat "$run_dir/python_preflight.log" >&2
  exit 1
fi

python3 - "$script_dir" "$run_dir" <<'PY'
import hashlib
import os
from pathlib import Path
import shutil
import sys

source_root, run_root = map(Path, sys.argv[1:])
snapshot = run_root / "source"
snapshot.mkdir()
files = []
for current, directories, names in os.walk(source_root, followlinks=False):
    directories[:] = sorted(name for name in directories
                             if name not in {"runs", "__pycache__", ".git"})
    current = Path(current)
    if any((current / name).is_symlink() for name in directories):
        raise ValueError("Source snapshots require regular files and directories; remove directory symlinks.")
    for name in sorted(names):
        original = current / name
        if original.is_symlink() or not original.is_file():
            raise ValueError("Source snapshots require regular files: " + str(original))
        relative = original.relative_to(source_root)
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        files.append((relative.as_posix(), hashlib.sha256(target.read_bytes()).hexdigest()))
with (run_root / "source_sha256.txt").open("w") as hashes:
    for relative, digest in sorted(files):
        hashes.write(digest + "  " + relative + "\n")
PY

python3 - "$run_dir/source" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from collect import validate_sources
validate_sources(Path(sys.argv[1]))
PY

printf 'CPU reference compile flags: -O2 -std=c++14\n' > "$run_dir/host_checks.log"
c++ -O2 -std=c++14 "$run_dir/source/test_reference.cpp" -o "$run_dir/test_reference" \
  >> "$run_dir/host_checks.log" 2>&1 || {
    cat "$run_dir/host_checks.log" >&2
    echo 'CPU reference checks failed to compile.' >&2
    exit 1
  }
"$run_dir/test_reference" >> "$run_dir/host_checks.log" 2>&1 || {
  cat "$run_dir/host_checks.log" >&2
  echo 'CPU reference self-check failed.' >&2
  exit 1
}
cat "$run_dir/host_checks.log"
nvcc -O2 -std=c++14 "$run_dir/source/device_info.cu" -o "$run_dir/device_info" \
  > "$run_dir/build.log" 2>&1 || {
    echo "CUDA discovery probe failed to compile. See $run_dir/build.log" >&2
    exit 1
  }
architecture="$("$run_dir/device_info" "$device" 2> "$run_dir/device_probe.log")" || {
  cat "$run_dir/device_probe.log" >&2
  exit 1
}
if [[ ! "$architecture" =~ ^[0-9]+$ ]]; then
  echo 'The CUDA device probe returned an invalid architecture.' >&2
  exit 1
fi
printf '%s\n' "$architecture" > "$run_dir/architecture.txt"
echo "Selected GPU architecture: sm_$architecture"
if nvcc --list-gpu-code > "$run_dir/nvcc_gpu_code.txt" 2> "$run_dir/nvcc_gpu_code_stderr.txt"; then
  if ! python3 - "$run_dir/nvcc_gpu_code.txt" "$architecture" <<'PY'
from pathlib import Path
import re
import sys
supported = set(re.findall(r"\bsm_[0-9]+[a-z]?\b", Path(sys.argv[1]).read_text()))
sys.exit(0 if "sm_" + sys.argv[2] in supported else 1)
PY
  then
    echo "This CUDA toolkit cannot target sm_$architecture. Load a compatible toolkit; diagnostics are retained in $run_dir." >&2
    exit 1
  fi
else
  echo 'nvcc cannot list GPU targets; checking the exact target by compiling it.' >> "$run_dir/build.log"
fi
printf 'Kernel compile flags: -O3 -std=c++14 -lineinfo -arch=sm_%s\n' "$architecture" >> "$run_dir/build.log"
nvcc -O3 -std=c++14 -lineinfo "-arch=sm_${architecture}" \
  "$run_dir/source/benchmark.cu" -o "$run_dir/benchmark" \
  >> "$run_dir/build.log" 2>&1 || {
    echo "Kernel compilation failed. This toolkit must support sm_$architecture. See $run_dir/build.log" >&2
    exit 1
  }
"$run_dir/benchmark" --selftest "$device" > "$run_dir/gpu_selftest.log" 2>&1 || {
  cat "$run_dir/gpu_selftest.log" >&2
  echo 'GPU correctness self-test failed; collection was not started.' >&2
  exit 1
}
cat "$run_dir/gpu_selftest.log"
set +e
python3 -u "$run_dir/source/collect.py" "$run_dir" "$mode" "$device" "${@:3}" \
  2> "$run_dir/progress.log" | tee "$run_dir/stdout.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
printf 'collector=%s\ntee=%s\n' "${pipeline_status[0]}" "${pipeline_status[1]}" > "$run_dir/collection_status.txt"
if (( pipeline_status[0] != 0 )); then
  cat "$run_dir/progress.log" >&2
  echo "Collection failed; logs and partial results are retained in $run_dir." >&2
fi
if (( pipeline_status[1] != 0 )); then
  echo "Live output capture failed with status ${pipeline_status[1]}; diagnostics are retained in $run_dir." >&2
fi
if (( pipeline_status[0] != 0 )); then exit "${pipeline_status[0]}"; fi
if (( pipeline_status[1] != 0 )); then exit "${pipeline_status[1]}"; fi
