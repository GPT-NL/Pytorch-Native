#!/bin/bash

set -e

export VENV_PATH="venv"
source $VENV_PATH/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export HF_HOME=/INSERT-GPT-NL-ROOT-DIR/.hf_cache_dir
HF_MODEL_PATH="allenai/OLMo-2-0325-32B@stage1-step118000-tokens990B" # comparable with 987B seen by GPT-NL step77440
# HF_MODEL_PATH="allenai/OLMo-2-0325-32B@stage1-step67441-tokens566B" # comparable with 561B seen by GPT-NL 44614 (epoch 1 end)
echo "HF_MODEL_PATH=$HF_MODEL_PATH"

TEMPERATURE=0.7 # 0.0 and 0.7
MAX_TOKENS=50

echo "Running inference for $HF_MODEL_PATH"


INPUT_FILE=$SCRIPT_DIR/inputs.json
OUTPUT_FILE_LOCAL="${SCRIPT_DIR}/outputs/olmo2_step118000_t_${TEMPERATURE}_max_tokens_${MAX_TOKENS}.json"
# OUTPUT_FILE_LOCAL="${SCRIPT_DIR}/outputs/olmo2_step67441_t_${TEMPERATURE}_max_tokens_${MAX_TOKENS}.json"

python $SCRIPT_DIR/inference_eval.py \
    --model $HF_MODEL_PATH --input-file $INPUT_FILE \
    --output-file $OUTPUT_FILE_LOCAL \
    --temperature $TEMPERATURE --max-tokens $MAX_TOKENS

echo "Finished eval"
