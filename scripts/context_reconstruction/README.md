# Context Reconstruction

Recipe for merging a short-context **BASE** checkpoint with a long-context **MID-TRAINED**
checkpoint, producing a single merged checkpoint ready for further training.

## Concept

Mid-training with short-context data degrades long-context ability inherited from
pretraining. Per **PRISM §8.1 / Granite-3.3**, linearly merging the BASE checkpoint
with the MID-TRAINED checkpoint (default **15% BASE / 85% MID-TRAINED**) before
long-context extension yields the strongest recovery at 128k while preserving
downstream reasoning performance. The approach is grounded in
[Model Soups (Wortsman et al., 2022)](https://arxiv.org/pdf/2203.05482) and
implemented here with [mergekit](https://github.com/arcee-ai/mergekit).

The merged output inherits the MID-TRAINED model's `max_position_embeddings`,
`rope_theta`, and `rope_scaling` so it can be further trained at the mid-trained
context length without changes.

## Install

This recipe uses the same environment of the project, no separate installation is needed. See the main [README.md](../../README.md) for how to install it.
(Edit `config.yaml` to point at valid model paths/IDs first — validation runs
before the merge, so `--dry-run` still needs them to resolve.) Subsequent
invocations — including the SLURM jobs — reuse the cached env and start
immediately.


**4. Output disk.** `output_dir` needs roughly 2× the mid-trained checkpoint
size — about 54 GB for a 27B bf16 model. The SLURM scripts default
`output_dir` to `/scratch-shared/$USER/...` via `config.yaml`; adjust if your
scratch quota is tight.

## Usage

Edit `config.yaml` with your BASE and MID-TRAINED paths and output location, then
submit the job:

CPU fallback (also set device: cpu in config.yaml)
### MAIN COMMAND
```
sbatch scripts/snellius/context_reconstruction.sh  # GPU (primary — recommended for 27B)
```

Override the output directory per submission:

```
sbatch scripts/snellius/context_reconstruction.sh --output-dir /scratch-shared/$USER/merged-exp42
```

Dry-run locally to inspect the resolved mergekit configuration without merging:

```
uv run --script scripts/context_reconstruction/merge_checkpoints.py \
    --config scripts/context_reconstruction/config.yaml --dry-run
```

Switch merge methods by editing `config.yaml`, e.g. TIES:

```yaml
method: ties
base_weight: 0.15
method_params:
  density: 0.5
  normalize: true
```

Supported methods: `linear` (default), `slerp`, `ties`, `dare_ties`, `task_arithmetic`, `sce`.

## Validate the merge

`validate_merge.py` checks the produced checkpoint on disk. The brief
`[check] OK ...` line printed by `merge_checkpoints.py` is unreliable when
YaRN is configured: transformers' loader mutates `max_position_embeddings`
to a 1e30 sentinel in memory and moves `rope_theta` under `rope_parameters`.
The on-disk `config.json` is intact regardless.

```
uv run --script scripts/context_reconstruction/validate_merge.py \
    --merged <merged_dir> --base <base_model_dir> --mid <mid_trained_dir> \
    --base-weight 0.15
```

Add `--kl` to also load all three checkpoints and compare next-token
distributions (~15 min extra on a single H100).

Steps: (1) NaN/Inf scan; (2) linear-combo on sampled tensors; (3) squared-norm
identity on every common tensor; (4) RMSNorm gain sanity; (5) tied-embedding
integrity; (6) `config.json` diff vs mid; (7) forward pass + greedy generation;
optional (8) KL vs parents.

Expected output for a healthy `0.85*mid + 0.15*base` linear merge (26B):

```
forward+gen   PASS — model.config.max_position_embeddings=32768, all logits finite
              "The capital of the Netherlands is" -> "Amsterdam."
              "Translate to Dutch: Good morning, how are you?" -> "Goedemorgen, hoe gaat het met u?"
kl            PASS — KL(merged || mid) ~0.02 nats, KL(merged || base) ~0.26 nats (5/5 closer to mid)
norm-id       PASS — worst residual ~5e-3 (bf16 noise floor)
linear-combo  PASS — sampled rel_err ~2e-3
```

## Output

The output directory contains sharded safetensors, the merged `config.json`
(with length/rope fields copied from MID-TRAINED), and the MID-TRAINED tokenizer —
ready to load via `AutoModelForCausalLM.from_pretrained(output_dir)`.

**Naming.** `output_dir` supports shell-style env vars (`$USER`, `${VAR}`, `~`)
and the following format placeholders, derived from the resolved config:

| Placeholder | Value |
|---|---|
| `{mid_name}` | basename of `mid_trained_model` |
| `{base_name}` | basename of `base_model` |
| `{method}` | merge method (`linear`, `slerp`, ...) |
| `{base_weight}` | `base_weight` value |
| `{dtype}` | `dtype` value |

The default config interpolates `{mid_name}-merged-{method}-b{base_weight}` so
each run lands in its own folder named after the MID-TRAINED checkpoint, e.g.
`/scratch-shared/$USER/<mid_basename>-merged-linear-b0.15`. Unknown placeholders
raise a clear error.

## References

- PRISM — https://arxiv.org/pdf/2603.17074
- Model Soups — https://arxiv.org/pdf/2203.05482
- mergekit — https://github.com/arcee-ai/mergekit
