#!/bin/bash
set -e

VENV_PATH="venv"
source $VENV_PATH/bin/activate

TOKENIZER_PATH="/INSERT-GPT-NL-TRAINING-DIR/tokenizer/gpt-nl-128k-1.0-newline-fix-v4"

JOB_COMMAND=$(scontrol show job "$SLURM_JOB_ID" | awk -F= '/Command=/ {print $2}')
echo JOB_COMMAND=$JOB_COMMAND
if [[ "$JOB_COMMAND" == *"(null)"* ]] || [[ "$JOB_COMMAND" == *"/bin/sh"* ]]; then
    echo "Not running under SLURM sbatch"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(realpath "$SCRIPT_DIR")"
else
    echo "Running under SLURM sbatch (job ID: $SLURM_JOB_ID)"
    # Get original jobscript path
    ORIG_DIR="$(dirname "$JOB_COMMAND")"
    SCRIPT_DIR=$ORIG_DIR
    REPO_DIR="$(realpath "$ORIG_DIR")"
fi
source $SCRIPT_DIR/../cli_helper.sh # contains get_converted_path

if [ "$#" -ne 1 ]; then
    echo "expecting CHECKPOINT_PATH from cli argument"
    exit 1
fi
CHECKPOINT_PATH=$1

# converted HF path
HF_MODEL_PATH=$(get_converted_path "$CHECKPOINT_PATH")
echo "Converting $CHECKPOINT_PATH to $HF_MODEL_PATH"
mkdir -p $HF_MODEL_PATH
SEQ_LENGTH=4096

if [ ! -d $HF_MODEL_PATH  ] || [ -z "$(ls -A $HF_MODEL_PATH)" ]; then
    echo "$HF_MODEL_PATH does not yet exist or is empty, converting checkpoint"
    python -m examples.huggingface.convert_checkpoint_to_hf -i $CHECKPOINT_PATH -o $HF_MODEL_PATH --dtype bfloat16 --device cpu

    # move old files if they exist
    for FILE in tokenizer.json tokenizer.model special_tokens_map.json tokenizer_config.json; do
        if [[ -e "$HF_MODEL_PATH/$FILE" ]]; then
            mv "$HF_MODEL_PATH/$FILE" "$HF_MODEL_PATH/$FILE.old"
        fi
    done
    # and place the fixed tokenizer
    ln -s $TOKENIZER_PATH/hf/* $HF_MODEL_PATH
else
    echo "$HF_MODEL_PATH already exists and is not empty, skipping conversion."
fi

echo "Finished conversion: $HF_MODEL_PATH"
