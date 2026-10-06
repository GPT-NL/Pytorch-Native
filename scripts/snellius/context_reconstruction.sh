#!/bin/bash -l
# SLURM job: merge BASE + MID-TRAINED checkpoints (GPU). Primary recipe for 27B models.
#SBATCH --job-name=context-reconstruction
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/context_reconstruction_%j.out

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TMPDIR=/scratch-shared/$USER

export HF_HOME=/scratch-shared/$USER/huggingface
export HF_DATASETS_CACHE=/scratch-shared/$USER/huggingface/datasets
export TRANSFORMERS_CACHE=/scratch-shared/$USER/huggingface/transformers
export HF_HUB_CACHE=/scratch-shared/$USER/huggingface/hub

mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_HUB_CACHE" logs

JOB_COMMAND=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/ {print $2}')
echo JOB_COMMAND=$JOB_COMMAND
if [[ -z "$SLURM_JOB_ID" ]] || [[ "$JOB_COMMAND" == *"(null)"* ]] || [[ "$JOB_COMMAND" == *"/bin/sh"* ]]; then
    echo "Not running under SLURM sbatch"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(realpath "$SCRIPT_DIR/../..")"
else
    echo "Running under SLURM sbatch (job ID: $SLURM_JOB_ID)"
    # Get original jobscript path
    ORIG_DIR="$(dirname "$JOB_COMMAND")"
    SCRIPT_DIR=$ORIG_DIR
    REPO_DIR="$(realpath "$ORIG_DIR/../..")"
fi

echo REPO_DIR=$REPO_DIR
exit 1

cd "$REPO_DIR"

echo "Starting context reconstruction merge (GPU)..."
echo "Working directory: $(pwd)"
echo ""

# Inline-deps script (PEP 723): mergekit resolves in its own isolated env, away
# from the main project's vllm/pydantic pin.
# pyproject.toml is updated to have mergekit
source venv/bin/activate
python scripts/context_reconstruction/merge_checkpoints.py \
    --config scripts/context_reconstruction/config.yaml "$@"

echo ""
echo "Context reconstruction merge completed!"
