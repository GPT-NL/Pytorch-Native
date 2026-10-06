#!/bin/bash

set -e

export VENV_PATH="venv_eval"
source $VENV_PATH/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# read HF_KEY here
source $SCRIPT_DIR/../snellius/.env

export CACHE_DIR=/INSERT-GPT-NL-ROOT-DIR/.hf_cache_dir # somewhere temporary not in home folder (will occupy some ~700GB)

# select branch to evaluate: https://huggingface.co/allenai/OLMo-2-0325-32B/tree/main
# HF_MODEL="allenai/OLMo-2-0325-32B@stage1-step0-tokens0B"
# HF_MODEL="allenai/OLMo-2-0325-32B@stage1-step1000-tokens9B"
# HF_MODEL="allenai/OLMo-2-0325-32B@stage1-step10000-tokens84B"
# HF_MODEL="allenai/OLMo-2-0325-32B@stage1-step41000-tokens344B" # add an option for comparison in july 2025
# HF_MODEL="allenai/OLMo-2-0325-32B@stage1-step61000-tokens512B"
HF_MODEL="allenai/OLMo-2-0325-32B@stage1-step67441-tokens566B" # comparable with 561B seen by GPT-NL 44614 (epoch 1 end)
# HF_MODEL="allenai/OLMo-2-0325-32B@stage1-step100000-tokens839B"
# HF_MODEL="allenai/OLMo-2-0325-32B@stage1-step118000-tokens990B" # comparable with 987B seen by GPT-NL step77440
# HF_MODEL="allenai/OLMo-2-0325-32B@stage1-step500000-tokens4195B"
# HF_MODEL="allenai/OLMo-2-0325-32B@stage1-step721901-tokens6056B" # final step for 6T tokens

STEP=$(echo "$HF_MODEL" | grep -oP '(?<=-step)\d+')
RUN_NAME=$HF_MODEL

echo "Running eval for $HF_MODEL step=$STEP run_name=$RUN_NAME"

python $SCRIPT_DIR/evaluate_checkpoint.py --model $HF_MODEL -l nl -l en --step $STEP --run_name $RUN_NAME --api_key $HF_KEY --cache_dir $CACHE_DIR #--task sentiment-classification

echo "Finished eval for $HF_MODEL"
