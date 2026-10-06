#!/bin/bash
set -e

module purge
# load python3.11 module so that venv is created with it
module load 2024 CUDA/12.6.0 Python/3.12.3-GCCcore-13.3.0


venv_path="${1:-venv}" # default venv name is venv, can be passed custom as CLI parameter
echo "installing to: $venv_path"
python -m venv $venv_path

echo "activating environment"
source $venv_path/bin/activate

echo "installing dependencies"
pip install -U pip setuptools wheel torch==2.7.0
pip install flash-attn==2.8.3 --no-build-isolation

echo "installing default packages"

echo "installing project"
pip install -e .[all]
