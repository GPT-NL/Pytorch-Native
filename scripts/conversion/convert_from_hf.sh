#!/bin/bash
set -e

VENV_PATH="venv"
source $VENV_PATH/bin/activate

TOKENIZER_PATH="/INSERT-GPT-NL-TRAINING-DIR/tokenizer/gpt-nl-128k-1.0-newline-fix-v4"
MAX_SEQ_LENGTH=32768
DEVICE="${DEVICE:-cpu}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "Usage: $0 <HF_CHECKPOINT_PATH> [OLMO_CORE_OUTPUT_PATH]"
    exit 1
fi

HF_CHECKPOINT_PATH="${1%/}"
if [ ! -d "$HF_CHECKPOINT_PATH" ]; then
    echo "Input does not exist: $HF_CHECKPOINT_PATH"
    exit 2
fi
if [ ! -f "$HF_CHECKPOINT_PATH/config.json" ] || ! compgen -G "$HF_CHECKPOINT_PATH/*.safetensors" > /dev/null; then
    echo "Input is not an HF checkpoint: $HF_CHECKPOINT_PATH"
    exit 3
fi

if [ "$#" -eq 2 ]; then
    OLMO_CORE_PATH="${2%/}"
else
    OLMO_CORE_PATH="$(dirname "$HF_CHECKPOINT_PATH")/$(basename "$HF_CHECKPOINT_PATH")-olmo-core"
fi

if [ -d "$OLMO_CORE_PATH" ] && [ -n "$(ls -A "$OLMO_CORE_PATH" 2>/dev/null)" ]; then
    echo "$OLMO_CORE_PATH already exists and is not empty, skipping."
    exit 0
fi
mkdir -p "$OLMO_CORE_PATH"

VALIDATE_FLAG=""
if [ "${VALIDATE:-0}" = "1" ]; then
    VALIDATE_FLAG="--validate"
fi

echo "Converting $HF_CHECKPOINT_PATH -> $OLMO_CORE_PATH"
python "$SCRIPT_DIR/convert_from_hf.py" \
    -i "$HF_CHECKPOINT_PATH" \
    -o "$OLMO_CORE_PATH" \
    -s "$MAX_SEQ_LENGTH" \
    --device "$DEVICE" \
    $VALIDATE_FLAG

if [ ! -e "$OLMO_CORE_PATH/tokenizer.json" ]; then
    ln -s "$TOKENIZER_PATH"/hf/* "$OLMO_CORE_PATH"
fi

# also create a .metadata file otherwise olmo-core thinks it's not a valid checkpoint
# olmo_core.train.checkpoint.Checkpointer.contains_checkpoint
echo -n '{"version": "2.1.0"}' > $OLMO_CORE_PATH/.metadata

echo "Finished conversion: $OLMO_CORE_PATH"
