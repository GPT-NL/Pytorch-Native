#!/bin/bash
#SBATCH --job-name=checkpoint-hf
# partition=gpu_h100: we don't really need GPUs - can work on fat_genoa or fat_rome
#SBATCH --partition=gpu_h100
#SBATCH --time=01:00:00 # lasts around 6 minutes
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

#TODO: still does not run on CPU-only node due to triton autotune context in moe.py

set -e

srun scripts/conversion/convert.sh $1
