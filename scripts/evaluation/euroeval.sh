#!/bin/bash

set -e

export VENV_PATH="venv_eval"
source $VENV_PATH/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -ne 1 ]; then
    echo "expecting CHECKPOINT_PATH from cli argument"
    exit 1
fi
source $SCRIPT_DIR/../cli_helper.sh # contains get_converted_path

if [ "$#" -ne 1 ]; then
    echo "expecting CHECKPOINT_PATH from cli argument"
    exit 1
fi
CHECKPOINT_PATH=$1
echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"

# make sure that it's the converted checkpoint path (or convert if needed)
HF_MODEL_PATH=$(get_converted_path "$CHECKPOINT_PATH")
if [ "$(realpath "$HF_MODEL_PATH")" = "$(realpath "$CHECKPOINT_PATH")" ]; then
    echo "CHECKPOINT_PATH passed points to converted model directory, conversion not needed"
else
    echo "CHECKPOINT_PATH is not in converted model directory, converting first"
    bash $SCRIPT_DIR/../conversion/convert.sh $CHECKPOINT_PATH
fi

echo "Running eval for $HF_MODEL_PATH"
# # If benchmark results file already exists, this will lead to silent failing. Deleting the old file instead
# if [ -f "$HF_MODEL_PATH/euroeval_benchmark_results.jsonl" ]; then
#     echo "Deleting existing euroeval_benchmark_results.jsonl from $HF_MODEL_PATH"
#     rm "$HF_MODEL_PATH/euroeval_benchmark_results.jsonl"
# fi

python $SCRIPT_DIR/evaluate_checkpoint.py --model $HF_MODEL_PATH -l nl -l en # --cache_dir $HF_HOME # --task sentiment-classification

echo "Finished eval for $HF_MODEL_PATH"
