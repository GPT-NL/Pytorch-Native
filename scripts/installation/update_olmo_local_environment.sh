#!/bin/bash
set -e

venv_path=venv

# activate the correct venv so that the destination of install will be correct
source ${venv_path}/bin/activate

# make sure to reinstall these ones just in case something changed
pip uninstall energy-utils ai2-olmo-core ai2-olmo-eval

pip install -U pip setuptools wheel torch==2.6.0
pip install flash-attn==2.7.4.post1 --no-build-isolation

# this will install olmo as non-editable mode, copying files from current working directory
pip install -e .[all]