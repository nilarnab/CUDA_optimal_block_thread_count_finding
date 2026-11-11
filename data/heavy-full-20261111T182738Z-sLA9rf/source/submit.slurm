#!/usr/bin/env bash
#SBATCH --job-name=lscat-heavy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=lscat-heavy-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
# Load a compatible CUDA/compiler/Python environment before sbatch.
bash run.sh "${MODE:-smoke}" 0
