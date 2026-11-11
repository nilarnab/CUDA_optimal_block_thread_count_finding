#!/usr/bin/env bash
#SBATCH --job-name=lscat-pilot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=lscat-%j.out
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit from the ls_cat_pilot directory}"
srun bash ./run.sh "${MODE:-smoke}" 0
