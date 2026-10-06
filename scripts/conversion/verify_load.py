"""Verify a converted olmo-core checkpoint loads and matches the HF source."""

from __future__ import annotations

# See convert_from_hf.py for the rationale on this triton stub.
import sys as _sys

_sys.modules.setdefault("triton", None)
_sys.modules.setdefault("triton.language", None)

import argparse
import json
import sys
from pathlib import Path

import torch
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.distributed.checkpoint import load_model_and_optim_state
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.utils import prepare_cli_environment
from safetensors import safe_open

# olmo-core key (Transformer.named_parameters), HF key (HF checkpoint state dict)
PROBES: list[tuple[str, str]] = [
    ("blocks.0.attention_norm.weight",   "model.layers.0.input_layernorm.weight"),
    ("blocks.10.attention.w_q.weight",   "model.layers.10.self_attn.q_proj.weight"),
    ("blocks.47.feed_forward.w3.weight", "model.layers.47.mlp.up_proj.weight"),
    ("embeddings.weight",                "model.embed_tokens.weight"),
    ("lm_head.w_out.weight",             "lm_head.weight"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--converted", type=Path, required=True)
    p.add_argument("--hf-source", type=Path, required=True)
    p.add_argument("--rtol", type=float, default=5e-3)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    converted: Path = args.converted
    hf_source: Path = args.hf_source

    if not (converted / "model_and_optim").is_dir():
        raise SystemExit(f"missing {converted}/model_and_optim/")
    if not (hf_source / "model.safetensors.index.json").is_file():
        raise SystemExit(f"missing {hf_source}/model.safetensors.index.json")

    tok = TokenizerConfig.gpt_nl()
    cfg = TransformerConfig.llama3_26B(vocab_size=tok.padded_vocab_size(), dtype=DType.bfloat16)
    log("[build] llama3_26B (bf16) on CPU")
    model = cfg.build()
    model.to_empty(device="cpu")

    log(f"[load]  {converted}/model_and_optim")
    load_model_and_optim_state(str(converted / "model_and_optim"), model)
    log("[load]  PASS")

    log("[finite] scanning parameters")
    n_params = 0
    bad: list[str] = []
    for name, p in model.named_parameters():
        n_params += p.numel()
        if not torch.isfinite(p).all().item():
            bad.append(name)
    if bad:
        log(f"[finite] FAIL — {len(bad)} non-finite tensors:")
        for n in bad[:10]:
            log(f"    {n}")
        sys.exit(1)
    log(f"[finite] PASS — {n_params / 1e9:.3f}B params finite")

    log("[probe] comparing tensors against HF source")
    with (hf_source / "model.safetensors.index.json").open() as f:
        hf_idx = json.load(f)["weight_map"]
    state_dict = dict(model.named_parameters())
    missing = [k for k, _ in PROBES if k not in state_dict]
    if missing:
        log(f"[probe] FAIL — probe keys absent: {missing}")
        sys.exit(1)

    header = f"  {'probe':<55s} {'shape':>20s} {'max_diff':>10s} {'cosine':>10s}  result"
    log(header)
    log("  " + "-" * (len(header) - 2))
    fail = False
    for olmo_key, hf_key in PROBES:
        olmo_t = state_dict[olmo_key].detach().float()
        with safe_open(str(hf_source / hf_idx[hf_key]), framework="pt", device="cpu") as fp:
            hf_t = fp.get_tensor(hf_key).float()
        if olmo_t.shape != hf_t.shape:
            log(f"  SHAPE MISMATCH {olmo_key}: {tuple(olmo_t.shape)} vs {tuple(hf_t.shape)}")
            fail = True
            continue
        diff_max = (olmo_t - hf_t).abs().max().item()
        cos = torch.nn.functional.cosine_similarity(
            olmo_t.flatten().unsqueeze(0), hf_t.flatten().unsqueeze(0)
        ).item()
        ref = max(hf_t.abs().max().item(), 1e-6)
        # cosine ~1 catches the case where rtol is tight but tensors are directionally identical
        ok = (diff_max < args.rtol * ref) or (cos > 0.9999)
        status = "PASS" if ok else "FAIL"
        log(f"  {olmo_key:<55s} {tuple(olmo_t.shape)!s:>20s} {diff_max:>10.2e} {cos:>10.6f}  {status}")
        if not ok:
            fail = True

    if fail:
        log("\n[overall] FAIL")
        sys.exit(1)
    log("\n[overall] PASS")


if __name__ == "__main__":
    prepare_cli_environment()
    main()
