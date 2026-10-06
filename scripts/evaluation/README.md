## Evaluation

This folder contains scripts for evaluation (NOT in-loop) using checkpoints converted to HuggingFace.

You need a different environment (`venv_eval` with different requirements from the training environment).
Install to `venv_eval` with `bash scripts/evaluation/install_eval.sh`.

## Evaluate a checkpoint

```bash
sbatch scripts/evaluation/run_euroeval.job <CHECKPOINT_PATH>
```
You can use both:
- `scripts/evaluation/run_euroeval.job <CHECKPOINT_PATH>` this is a slurm job declaration which will call the `scripts/evaluation/euroeval.sh`
- directly the `scripts/evaluation/euroeval.sh <CHECKPOINT_PATH>` if you don't use slurm or you are inside an interactive job

For `<CHECKPOINT_PATH>` you can use both OLMo checkpoints or converted Hugginface Checkpoints:
1. if `<CHECKPOINT_PATH>` points at OLMo checkpoint (e.g. inside `/INSERT-GPT-NL-TRAINING-DIR/OLMo-core/`), the script will first check if a converted checkpoint exists and convert it
2. if `<CHECKPOINT_PATH>` instead points at converted checkpoint (inside `/INSERT-GPT-NL-TRAINING-DIR/converted_checkpoints/`), only evaluation is run

## Evaluate OLMo-32B model

<!--
TODO: Refer to the GPT-NL evaluation report when that is released.
-->
We compare our training with the one of OLMo-32B, with the checkpoints provided on [HuggingFace](https://huggingface.co/allenai/OLMo-2-0325-32B/tree/main).

The script to run euroeval against these checkpoints is `scripts/evaluation/euroeval_olmo.sh`. Uncomment appropriately the `$HF_MODEL` variable to select the proper step.

