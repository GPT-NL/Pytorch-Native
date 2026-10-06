#!/bin/bash
#SBATCH --job-name=checkpoint-from-hf
#SBATCH --partition=genoa
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=300G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -e

scripts/conversion/convert_from_hf.sh "$@"
