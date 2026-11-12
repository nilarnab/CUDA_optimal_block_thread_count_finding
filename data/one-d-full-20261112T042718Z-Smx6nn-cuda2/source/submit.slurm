#!/usr/bin/env bash
#SBATCH --job-name=lscat-1d
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=lscat-1d-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Load a compatible CUDA/compiler/Python environment before sbatch.
bash run.sh "${MODE:-smoke}" 0 --seeds "${SEEDS:-1}" --repeats "${REPEATS:-3}"
