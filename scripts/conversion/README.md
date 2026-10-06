# Checkpoint conversion

## OLMo-core → HuggingFace

`convert.sh` takes an OLMo-core checkpoint and converts it to HuggingFace
safetensors. Use either:

- `scripts/conversion/run_convert.sh <CHECKPOINT_PATH>` — Slurm wrapper that
  calls `convert.sh`.
- `scripts/conversion/convert.sh <CHECKPOINT_PATH>` — direct (interactive job or
  no Slurm).

`CHECKPOINT_PATH` is something like
`/INSERT-GPT-NL-TRAINING-DIR/OLMo-core/gpt-nl-26B-nodes-22-mbs-12-gas-3-12891307/step18040`.

## HuggingFace → OLMo-core

`convert_from_hf.sh` is the reverse direction, for feeding a merged HF
checkpoint (e.g. from `scripts/context_reconstruction/`) back into the
mid-training loop.

```
sbatch scripts/conversion/run_convert_from_hf.sh <HF_CHECKPOINT_PATH>
# or:
scripts/conversion/convert_from_hf.sh <HF_CHECKPOINT_PATH> [OUTPUT_PATH]
```

Output defaults to `<hf_parent>/<hf_basename>-olmo-core/`. The wrapper uses the
same `TransformerConfig.llama3_26B` and `TokenizerConfig.gpt_nl` factories as
`scripts/train_context.py`, and pins `--max-sequence-length 32768` to match
`config/gpt-nl-26B-mid-train.yaml`.

The Python helper stubs `triton` before importing `olmo_core` so the
`@triton.autotune` decorators in `olmo_core.kernels.moe` do not run their
driver-init scan on CPU-only nodes (`olmo_core.ops.moe` already handles the
resulting ImportError; Llama does not invoke the MoE kernels). It also
overwrites `config.json` after the upstream call returns, because upstream
writes via an unflushed `NamedTemporaryFile` and the destination ends up empty.

Runs on CPU by default; the bf16 build holds peak RSS around 150 GB. Set
`DEVICE=cuda` only on a node with sufficient GPU memory. Set `VALIDATE=1` to
enable the upstream logit-match validator (loads a second copy of the HF model
in fp32; OOMs a single H100).

**Optimizer/scheduler state is not preserved.** HF safetensors do not carry
those slots. The trainer's `LoadStrategy.if_available` loads model weights only
and starts optimizer/scheduler fresh — the expected behavior for the PRISM §8.1
context-reconstruction recipe.

Launch mid-training from the converted checkpoint:

```
export LOAD_PATH=<converted_olmo_core_dir>
sbatch scripts/snellius/launch_mid_train_job.job
```

`mid_train.sh` only auto-populates `LOAD_PATH` when it is empty, so this export
is honored.

### Verifying the converted checkpoint

`verify_load.py` builds the trainer's `TransformerConfig.llama3_26B` model in
bf16, loads the DCP shards via `load_model_and_optim_state`, scans all
parameters for non-finite values, and diffs five probe tensors (embeddings,
lm_head, early/mid/late layer weights) against the HF source.

```
sbatch scripts/conversion/run_verify_load.sh [<converted_dir>] [<hf_source_dir>]
```

A healthy conversion reports `[finite] PASS` and `max_diff = 0` on every probe
(bit-identical with the HF source). Forward-pass smoke output with
`use_flash=True` matches the HF source within bf16 noise:

```
"The capital of the Netherlands is"                       -> "Amsterdam"
"Translate to Dutch: Good morning, how are you?\nDutch:"  -> "Goed"
```

Note: PyTorch's SDPA fallback (`use_flash=False`) produces all-NaN logits at
26 B / bf16 — a numerical issue in that path, not a checkpoint defect. The
trainer's `use_flash_attention_2: true` setting avoids it.
