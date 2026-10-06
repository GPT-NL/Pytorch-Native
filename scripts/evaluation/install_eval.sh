#!/bin/bash


# euroeval last release does not support flash_attn
# >> The `flash_attn` package is not supported by EuroEval, as it is now built into the other packages and it conflicts with the other implementations. Please uninstall it using `pip uninstall flash_attn` and try again.
# we need a separate environment

set -e

module purge
# load python3.11 module so that venv is created with it
module load 2024 CUDA/12.6.0 Python/3.12.3-GCCcore-13.3.0


venv_path="${1:-venv_eval}" # default venv name is venv_eval, can be passed custom as CLI parameter
echo "installing to: $venv_path"
python -m venv $venv_path

echo "activating environment"
source $venv_path/bin/activate

echo "installing dependencies"
pip install -U pip setuptools wheel

echo "installing default packages"

echo "installing project"
pip install -e .[evaluation]