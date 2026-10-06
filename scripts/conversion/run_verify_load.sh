#!/bin/bash
#SBATCH --job-name=verify-converted
#SBATCH --partition=genoa
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

set -e
source venv/bin/activate

CONVERTED="${1:-/INSERT-GPT-NL-TRAINING-DIR/merged_checkpoints/step146084-merged-linear-b0.15-olmo-core}"
HF_SOURCE="${2:-/INSERT-GPT-NL-TRAINING-DIR/merged_checkpoints/step146084-merged-linear-b0.15}"

python scripts/conversion/verify_load.py \
    --converted "$CONVERTED" \
    --hf-source "$HF_SOURCE"
