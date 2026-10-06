#!/bin/bash

set -e

export VENV_PATH="venv"
source $VENV_PATH/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
    source $SCRIPT_DIR/../conversion/convert.sh $CHECKPOINT_PATH
fi

TEMPERATURE=0.7 # 0.0 and 0.7
MAX_TOKENS=50

echo "Running inference for $HF_MODEL_PATH"


INPUT_FILE=$SCRIPT_DIR/inputs.json
OUTPUT_FILE_MODEL_DIR="${HF_MODEL_PATH}/inference_outputs_t_${TEMPERATURE}_max_tokens_${MAX_TOKENS}.json"


# create file name that contains run name, step and all important info
PATH_WITHOUT_TRAILING_SLASH="${HF_MODEL_PATH%/}"
STEP_STR="${PATH_WITHOUT_TRAILING_SLASH##*/}"
REST="${PATH_WITHOUT_TRAILING_SLASH%/*}"
RUN_NAME="${REST##*/}"
NORMALIZED_FILE_NAME="$RUN_NAME/$STEP_STR"
NORMALIZED_FILE_NAME="${NORMALIZED_FILE_NAME//\//_}"
OUTPUT_FILE_LOCAL="${SCRIPT_DIR}/outputs/${NORMALIZED_FILE_NAME}_t_${TEMPERATURE}_max_tokens_${MAX_TOKENS}.json"
echo $OUTPUT_FILE_LOCAL

python $SCRIPT_DIR/inference_eval.py \
    --model $HF_MODEL_PATH --input-file $INPUT_FILE \
    --output-file $OUTPUT_FILE_LOCAL \
    --temperature $TEMPERATURE --max-tokens $MAX_TOKENS

# also copy in model directory (could fail for permissions)
cp $OUTPUT_FILE_LOCAL $OUTPUT_FILE_MODEL_DIR
chmod g+rw $OUTPUT_FILE_MODEL_DIR

echo "Finished eval"
