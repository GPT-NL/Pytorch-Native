# Inference

This folder contains some scripts and examples to run the GPT-NL model for inference.

## Run inference

See the file `example_inference.py` to see how to load the model and generate text.

## Evaluate inference

Use the `scripts/inference/inputs.json` and evaluate a converted model with:

```bash
# in interactive job
./scripts/inference/inference.sh <CHECKPOINT_PATH>
# or with the generic script
sbatch scripts/run_generic_h100.job ./scripts/inference/inference.sh <CHECKPOINT_PATH>
```

The output text is saved at the converted checkpoint location and copied to the local `scripts/inference/outputs` folder