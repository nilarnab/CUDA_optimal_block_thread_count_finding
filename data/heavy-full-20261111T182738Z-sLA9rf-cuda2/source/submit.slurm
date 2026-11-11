#!/usr/bin/env bash
#SBATCH --job-name=lscat-heavy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=lscat-heavy-%j.out
#SBATCH --partition=c12m85-a100-1
#SBATCH --account=csci_ga_3033_025-2026fa

singularity exec --nv \
--overlay /scratch/sm12779/overlay-15GB-500K.ext3:ro \
/scratch/sm12779/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
/bin/bash -c "source /ext3/env.sh; which python; ./run.sh full 0"
