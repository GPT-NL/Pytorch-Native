# /// script
# requires-python = ">=3.12"
# dependencies = [
#     # Pinned to 0.0.x: mergekit 0.1.x has a pydantic forward-reference bug in
#     # ConfiguredModuleArchitecture (model_rebuild() is never called after torch
#     # is imported, so instantiation raises PydanticUserError "class-not-fully-defined").
#     # 0.0.6 uses the older architecture code path and works correctly for all
#     # methods exposed by this driver.
#     "mergekit>=0.0.6,<0.1",
#     "pyyaml>=6.0",
# ]
# ///
"""Merge a BASE checkpoint with a MID-TRAINED checkpoint for long-context restoration.

Implements the PRISM §8.1 recipe: linearly merge a short-context BASE model with a
mid-trained model (default 15% / 85%) before long-context extension. Supports
linear, slerp, ties, dare_ties, task_arithmetic, and sce via mergekit.

The merged output inherits the MID-TRAINED model's positional / rope config, so it is
ready for further long-context training at the mid-trained length.

Run with `uv run --script scripts/context_reconstruction/merge_checkpoints.py ...`
so that mergekit's pinned dependencies (pydantic) resolve in their own env, isolated
from the main project's vllm/pydantic pin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml

PATH_KEYS: tuple[str, ...] = ("base_model", "mid_trained_model", "output_dir")

SUPPORTED_METHODS = {"linear", "slerp", "ties", "dare_ties", "task_arithmetic", "sce"}
SUPPORTED_DTYPES = {"bfloat16", "float16", "float32"}
SUPPORTED_DEVICES = {"cuda", "cpu"}

# Config fields propagated from MID-TRAINED into the merged output after mergekit writes
# its own config.json. Guards against the output silently inheriting BASE's short-context
# settings if mergekit's tokenizer/config-source resolution changes in future versions.
MID_CONFIG_KEYS: tuple[str, ...] = (
    "max_position_embeddings",
    "rope_theta",
    "rope_scaling",
    "sliding_window",
    "use_sliding_window",
)
TOKENIZER_FILES: tuple[str, ...] = (
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge a BASE checkpoint with a MID-TRAINED checkpoint (mergekit).")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output_dir from the YAML (useful for SLURM $SLURM_JOB_ID templating)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the mergekit configuration, print it as YAML, and exit without merging",
    )
    return p.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise SystemExit(f"[config] {path} did not parse as a YAML mapping")
    return cfg


def _expand_paths(cfg: dict[str, Any]) -> None:
    """Expand `$VAR`, `${VAR}` and `~` in path-like fields in place."""
    for key in PATH_KEYS:
        val = cfg.get(key)
        if isinstance(val, str):
            cfg[key] = os.path.expanduser(os.path.expandvars(val))


def _format_output_dir(cfg: dict[str, Any]) -> None:
    """Substitute {mid_name}/{base_name}/{method}/{base_weight}/{dtype} in output_dir."""
    output_dir = cfg.get("output_dir")
    if not isinstance(output_dir, str) or "{" not in output_dir:
        return
    placeholders = {
        "base_name": Path(str(cfg.get("base_model", "")).rstrip("/")).name,
        "mid_name": Path(str(cfg.get("mid_trained_model", "")).rstrip("/")).name,
        "method": cfg.get("method", "linear"),
        "base_weight": cfg.get("base_weight", 0.15),
        "dtype": cfg.get("dtype", "bfloat16"),
    }
    try:
        cfg["output_dir"] = output_dir.format(**placeholders)
    except KeyError as e:
        raise SystemExit(f"[config] unknown placeholder in output_dir: {e}")


def _looks_like_local_or_hub(ref: str) -> bool:
    if Path(ref).exists():
        return True
    return not ref.startswith(("/", "./", "~")) and ref.count("/") >= 1


def _validate_config(cfg: dict[str, Any]) -> None:
    required = ("base_model", "mid_trained_model", "output_dir", "method", "dtype", "device")
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(f"[config] missing required keys: {missing}")

    if cfg["method"] not in SUPPORTED_METHODS:
        raise SystemExit(f"[config] method must be one of {sorted(SUPPORTED_METHODS)}, got {cfg['method']!r}")
    if cfg["dtype"] not in SUPPORTED_DTYPES:
        raise SystemExit(f"[config] dtype must be one of {sorted(SUPPORTED_DTYPES)}, got {cfg['dtype']!r}")
    if cfg["device"] not in SUPPORTED_DEVICES:
        raise SystemExit(f"[config] device must be one of {sorted(SUPPORTED_DEVICES)}, got {cfg['device']!r}")

    bw = float(cfg.get("base_weight", 0.15))
    if not 0.0 <= bw <= 1.0:
        raise SystemExit(f"[config] base_weight must be in [0, 1], got {bw}")

    for key in ("base_model", "mid_trained_model"):
        if not _looks_like_local_or_hub(cfg[key]):
            raise SystemExit(f"[config] {key}={cfg[key]!r} is neither an existing local path nor a hub id")

    mp = cfg.get("method_params")
    if mp is not None and not isinstance(mp, dict):
        raise SystemExit("[config] method_params must be a mapping")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_tokenizer_compat(base: str, mid: str, trust_remote_code: bool) -> None:
    """Hard-fail on vocab-size divergence; warn on tokenizer.json byte-level divergence."""
    from transformers import AutoTokenizer

    log("[check] loading tokenizers for compatibility check...")
    tok_base = AutoTokenizer.from_pretrained(base, trust_remote_code=trust_remote_code, use_fast=True)
    tok_mid = AutoTokenizer.from_pretrained(mid, trust_remote_code=trust_remote_code, use_fast=True)
    if tok_base.vocab_size != tok_mid.vocab_size:
        raise SystemExit(
            f"[check] tokenizer vocab_size mismatch: base={tok_base.vocab_size}, "
            f"mid={tok_mid.vocab_size}. Merging across divergent embedding tables is unsafe."
        )

    base_tok = Path(base) / "tokenizer.json"
    mid_tok = Path(mid) / "tokenizer.json"
    if base_tok.exists() and mid_tok.exists() and _sha256(base_tok) != _sha256(mid_tok):
        log(
            "[check] tokenizer.json bytes differ but vocab_size matches; "
            "proceeding — added_tokens/merges may differ"
        )
    log(f"[check] tokenizer compatibility OK (vocab_size={tok_mid.vocab_size})")


def build_merge_dict(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build a mergekit-compatible config dict, mid-trained always listed first.

    The MID-TRAINED model is set as `base_model` for every method so mergekit inherits
    its config.json + tokenizer into the output. `post_merge_fixup` then double-checks
    the critical length/rope fields.
    """
    method = cfg["method"]
    base = cfg["base_model"]
    mid = cfg["mid_trained_model"]
    base_weight = float(cfg.get("base_weight", 0.15))
    mid_weight = 1.0 - base_weight
    dtype = cfg["dtype"]
    extra = dict(cfg.get("method_params") or {})

    if method == "linear":
        merge = {
            "merge_method": "linear",
            "base_model": mid,
            "models": [
                {"model": mid, "parameters": {"weight": mid_weight}},
                {"model": base, "parameters": {"weight": base_weight}},
            ],
            "parameters": {"normalize": bool(extra.pop("normalize", True))},
            "dtype": dtype,
        }
    elif method == "slerp":
        t = float(extra.pop("t", 0.5))
        merge = {
            "merge_method": "slerp",
            "base_model": mid,
            "models": [{"model": mid}, {"model": base}],
            "parameters": {"t": t},
            "dtype": dtype,
        }
    elif method in ("ties", "dare_ties"):
        density = float(extra.pop("density", 0.5))
        merge = {
            "merge_method": method,
            "base_model": mid,
            "models": [
                {"model": base, "parameters": {"weight": base_weight, "density": density}},
            ],
            "parameters": {"normalize": bool(extra.pop("normalize", True))},
            "dtype": dtype,
        }
    elif method == "task_arithmetic":
        merge = {
            "merge_method": "task_arithmetic",
            "base_model": mid,
            "models": [{"model": base, "parameters": {"weight": base_weight}}],
            "parameters": {"normalize": bool(extra.pop("normalize", True))},
            "dtype": dtype,
        }
    elif method == "sce":
        merge = {
            "merge_method": "sce",
            "base_model": mid,
            "models": [
                {"model": mid, "parameters": {"weight": mid_weight}},
                {"model": base, "parameters": {"weight": base_weight}},
            ],
            "parameters": {
                "select_topk": float(extra.pop("select_topk", 1.0)),
                "int8_mask": bool(extra.pop("int8_mask", False)),
            },
            "dtype": dtype,
        }
    else:
        raise SystemExit(f"[merge] unsupported method: {method}")

    if extra:
        merge["parameters"].update(extra)
    return merge


def _patch_mergekit_forward_refs() -> None:
    """Inject torch into mergekit.architecture's namespace and rebuild affected models.

    mergekit's ConfiguredArchitectureInfo has a `config: PretrainedConfig` field;
    pydantic 2.10's stricter introspection walks PretrainedConfig's annotations,
    encounters a `torch.*` reference, and demands `torch` be resolvable in the
    defining module's globals. mergekit's architecture.py doesn't import torch,
    so instantiation raises PydanticUndefinedAnnotation. Injecting torch and
    calling model_rebuild(force=True) resolves the forward reference cleanly.
    """
    import torch
    from mergekit import architecture as mk_arch

    mk_arch.torch = torch
    mk_arch.ConfiguredArchitectureInfo.model_rebuild(force=True)


def run_mergekit(merge_dict: dict[str, Any], output_dir: Path, cfg: dict[str, Any]) -> None:
    from mergekit.config import MergeConfiguration
    from mergekit.merge import run_merge
    from mergekit.options import MergeOptions

    _patch_mergekit_forward_refs()

    merge_cfg = MergeConfiguration.model_validate(merge_dict)
    options = MergeOptions(
        cuda=(cfg["device"] == "cuda"),
        low_cpu_memory=bool(cfg.get("low_cpu_memory", True)),
        lazy_unpickle=bool(cfg.get("lazy_unpickle", True)),
        trust_remote_code=bool(cfg.get("trust_remote_code", False)),
        copy_tokenizer=True,
        random_seed=int(cfg.get("seed", 0)),
    )
    log(f"[merge] running mergekit method={cfg['method']} device={cfg['device']}")
    run_merge(merge_cfg, str(output_dir), options)
    log(f"[merge] wrote merged checkpoint to {output_dir}")


def post_merge_fixup(output_dir: Path, mid: str, trust_remote_code: bool) -> None:
    """Overwrite length/rope fields in config.json with the MID-TRAINED values; copy tokenizer."""
    mid_path = Path(mid)
    if mid_path.exists():
        with open(mid_path / "config.json") as f:
            mid_cfg_dict = json.load(f)
    else:
        from transformers import AutoConfig

        mid_cfg_dict = AutoConfig.from_pretrained(mid, trust_remote_code=trust_remote_code).to_dict()

    out_cfg_path = output_dir / "config.json"
    with open(out_cfg_path) as f:
        out_cfg = json.load(f)

    changed: list[str] = []
    for key in MID_CONFIG_KEYS:
        if key in mid_cfg_dict and out_cfg.get(key) != mid_cfg_dict[key]:
            out_cfg[key] = mid_cfg_dict[key]
            changed.append(key)
    if changed:
        with open(out_cfg_path, "w") as f:
            json.dump(out_cfg, f, indent=2)
        log(f"[fixup] config.json overwritten from mid-trained: {changed}")
    else:
        log("[fixup] config.json already matches mid-trained for length/rope fields")

    if mid_path.exists():
        for name in TOKENIZER_FILES:
            src = mid_path / name
            dst = output_dir / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
                log(f"[fixup] copied {name} from mid-trained")


def sanity_check(output_dir: Path, dtype: str, trust_remote_code: bool) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    log(f"[check] reloading merged model from {output_dir} for sanity check...")
    out_cfg = AutoConfig.from_pretrained(output_dir, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        output_dir,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=trust_remote_code,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(
        f"[check] OK — max_position_embeddings={out_cfg.max_position_embeddings}, "
        f"rope_theta={getattr(out_cfg, 'rope_theta', None)}, n_params={n_params / 1e9:.2f}B"
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    _expand_paths(cfg)
    _format_output_dir(cfg)
    _validate_config(cfg)

    output_dir = Path(cfg["output_dir"])
    merge_dict = build_merge_dict(cfg)

    if args.dry_run:
        log("[dry-run] resolved mergekit configuration:")
        print(yaml.safe_dump(merge_dict, sort_keys=False))
        return

    check_tokenizer_compat(
        cfg["base_model"],
        cfg["mid_trained_model"],
        bool(cfg.get("trust_remote_code", False)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_mergekit(merge_dict, output_dir, cfg)
    post_merge_fixup(output_dir, cfg["mid_trained_model"], bool(cfg.get("trust_remote_code", False)))

    if not bool(cfg.get("skip_sanity_check", False)):
        sanity_check(output_dir, cfg["dtype"], bool(cfg.get("trust_remote_code", False)))


if __name__ == "__main__":
    main()
