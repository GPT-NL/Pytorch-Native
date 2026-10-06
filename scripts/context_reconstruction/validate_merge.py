# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch",
#     "transformers>=4.50,<5.10",
#     "safetensors>=0.4",
#     "accelerate>=0.30",
# ]
# ///
"""Validate a context-reconstruction merged checkpoint produced by merge_checkpoints.py.

Checks the on-disk artifact directly. The brief ``[check] OK ...`` line in
merge_checkpoints.py is unreliable when YaRN is configured: transformers'
``AutoConfig.from_pretrained`` mutates ``max_position_embeddings`` to a 1e30
sentinel and moves ``rope_theta`` under ``rope_parameters``. This script reads
the on-disk config and the loaded model directly.

Default checks (~5 min, single GPU):
  1. NaN/Inf scan across all safetensors shards.
  2. Linear-combo on a sample of tensors: merged ~= (1-bw)*mid + bw*base.
  3. Squared-norm identity on every common tensor.
  4. LayerNorm/RMSNorm gain sanity (dead channels, magnitude drift).
  5. Tied-embedding integrity vs config.tie_word_embeddings.
  6. config.json diff against the mid-trained source.
  7. Forward pass + greedy generation on a few prompts.

Optional (~15 min, --kl):
  8. KL(merged || mid) vs KL(merged || base) on a small prompt set.

Usage:
  uv run --script scripts/context_reconstruction/validate_merge.py \\
      --merged <merged_dir> --base <base_dir> --mid <mid_dir> --base-weight 0.15
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def log(msg: str) -> None:
    print(msg, flush=True)


def hr(title: str) -> None:
    log("=" * 78)
    log(title)
    log("=" * 78)


def open_indexed(dir_: Path) -> dict:
    idx_path = dir_ / "model.safetensors.index.json"
    if idx_path.exists():
        with idx_path.open() as f:
            idx = json.load(f)
        return {k: dir_ / v for k, v in idx["weight_map"].items()}
    cands = list(dir_.glob("*.safetensors"))
    if len(cands) != 1:
        raise SystemExit(f"expected sharded index or single safetensors in {dir_}, got {len(cands)}")
    only = cands[0]
    with safe_open(str(only), framework="pt", device="cpu") as fp:
        return {k: only for k in fp}


def get_tensor(idx_map: dict, key: str) -> torch.Tensor:
    p = idx_map[key]
    with safe_open(str(p), framework="pt", device="cpu") as fp:
        return fp.get_tensor(key)


def check_finiteness(merged_dir: Path) -> bool:
    idx = open_indexed(merged_dir)
    shards = sorted(set(idx.values()))
    log(f"[stats] scanning {len(idx)} tensors across {len(shards)} shards")
    dtype_counts: dict[str, int] = {}
    total_params = 0
    bad: list[tuple[str, int, int]] = []
    for shard in shards:
        keys_in_shard = [k for k, v in idx.items() if v == shard]
        nan = inf = params = 0
        with safe_open(str(shard), framework="pt", device="cpu") as fp:
            for k in keys_in_shard:
                t = fp.get_tensor(k)
                f = t.float()
                n_nan = int(torch.isnan(f).sum().item())
                n_inf = int(torch.isinf(f).sum().item())
                nan += n_nan
                inf += n_inf
                params += t.numel()
                dtype_counts[str(t.dtype)] = dtype_counts.get(str(t.dtype), 0) + 1
                if n_nan or n_inf:
                    bad.append((k, n_nan, n_inf))
        total_params += params
        log(f"  {shard.name}: tensors={len(keys_in_shard)}, params={params / 1e9:.3f}B, nan={nan}, inf={inf}")
    log(f"[stats] total params: {total_params / 1e9:.3f}B   dtype: {dtype_counts}")
    if bad:
        log(f"[stats] FAIL: {len(bad)} tensors have non-finite values:")
        for k, n_nan, n_inf in bad[:10]:
            log(f"    {k}: nan={n_nan}, inf={n_inf}")
        return False
    log("[stats] PASS")
    return True


def check_linear_combo(
    merged_dir: Path,
    mid_dir: Path,
    base_dir: Path,
    base_weight: float,
    n_samples: int = 12,
    rtol: float = 5e-3,
) -> bool:
    # rtol is loose because bf16 has a 7-bit mantissa and the weighted sum is
    # stored back in bf16, introducing ~1e-3 relative error per element.
    w_m, w_b = 1.0 - base_weight, base_weight
    log(f"[combo] merged ~= {w_m:.3f}*mid + {w_b:.3f}*base on {n_samples} tensors (rtol={rtol})")
    merged_idx = open_indexed(merged_dir)
    mid_idx = open_indexed(mid_dir)
    base_idx = open_indexed(base_dir)
    common = sorted(set(merged_idx) & set(mid_idx) & set(base_idx))
    log(f"[combo] {len(common)} tensors common to all three checkpoints")
    rng = random.Random(0)
    sample = rng.sample(common, min(n_samples, len(common)))
    failed = 0
    for k in sample:
        m = get_tensor(merged_idx, k).float()
        a = get_tensor(mid_idx, k).float()
        b = get_tensor(base_idx, k).float()
        if m.shape != a.shape or m.shape != b.shape:
            log(f"  SHAPE MISMATCH {k}: merged={tuple(m.shape)} mid={tuple(a.shape)} base={tuple(b.shape)}")
            failed += 1
            continue
        expected = w_m * a + w_b * b
        diff = (m - expected).abs()
        ref = expected.abs().mean().clamp_min(1e-8)
        rel = (diff.mean() / ref).item()
        ok = rel < rtol
        status = "ok " if ok else "FAIL"
        log(f"  [{status}] {k}: shape={tuple(m.shape)} rel_err={rel:.2e} abs_max_err={diff.max().item():.2e}")
        if not ok:
            failed += 1
    if failed:
        log(f"[combo] FAIL: {failed}/{len(sample)} tensors violate linear-combo")
        return False
    log("[combo] PASS")
    return True


def check_norm_identity(
    merged_dir: Path,
    mid_dir: Path,
    base_dir: Path,
    base_weight: float,
    rtol: float = 0.02,
) -> bool:
    # ||m||^2 ~= w_m^2 * ||a||^2 + 2 * w_m * w_b * <a, b> + w_b^2 * ||b||^2
    w_m, w_b = 1.0 - base_weight, base_weight
    merged_idx = open_indexed(merged_dir)
    mid_idx = open_indexed(mid_dir)
    base_idx = open_indexed(base_dir)
    common = sorted(set(merged_idx) & set(mid_idx) & set(base_idx))
    log(f"[norm-id] checking {len(common)} tensors (rtol={rtol})")
    fails: list[tuple[str, float]] = []
    residuals: list[tuple[str, float, float, float]] = []
    for k in common:
        m = get_tensor(merged_idx, k).float()
        a = get_tensor(mid_idx, k).float()
        b = get_tensor(base_idx, k).float()
        if m.shape != a.shape or m.shape != b.shape:
            fails.append((k, float("inf")))
            continue
        nm = (m * m).sum().item()
        na = (a * a).sum().item()
        nb = (b * b).sum().item()
        dot = (a * b).sum().item()
        pred = (w_m * w_m) * na + 2.0 * w_m * w_b * dot + (w_b * w_b) * nb
        rel = abs(nm - pred) / max(pred, 1e-12)
        residuals.append((k, nm**0.5, max(pred, 0.0) ** 0.5, rel))
        if rel > rtol:
            fails.append((k, rel))
    if fails:
        log(f"[norm-id] FAIL: {len(fails)} tensors exceed rtol")
        for k, r in sorted(fails, key=lambda x: -x[1])[:10]:
            log(f"    {k}: rel_err={r:.2e}")
        return False
    log("[norm-id] PASS — worst residuals:")
    for k, mn, pn, r in sorted(residuals, key=lambda x: -x[3])[:5]:
        log(f"    {k}: merged_norm={mn:.3f} predicted={pn:.3f} rel_err={r:.2e}")
    return True


def check_layernorm_gains(merged_dir: Path, mid_dir: Path) -> bool:
    # Negative gains are valid in pretrained Llama-class models; only flag
    # near-zero values (dead channels) and large magnitude drift vs mid.
    merged_idx = open_indexed(merged_dir)
    mid_idx = open_indexed(mid_dir)
    keys = sorted(
        k for k in merged_idx if "layernorm" in k.lower() or k.endswith(".norm.weight") or k == "model.norm.weight"
    )
    log(f"[ln] checking {len(keys)} layernorm tensors")
    bad: list[str] = []
    n_have_negatives = 0
    for k in keys:
        m = get_tensor(merged_idx, k).float()
        n_dead = int((m.abs() < 1e-6).sum().item())
        n_neg = int((m < 0).sum().item())
        if n_neg:
            n_have_negatives += 1
        if k in mid_idx:
            a = get_tensor(mid_idx, k).float()
            ratio = (m.abs().mean() / a.abs().mean().clamp_min(1e-8)).item()
            if not (0.5 < ratio < 2.0):
                bad.append(f"{k}: mean_mag_ratio_vs_mid={ratio:.3f}")
        if n_dead:
            bad.append(f"{k}: dead_channels={n_dead}")
    log(f"[ln] {n_have_negatives}/{len(keys)} tensors contain some negative gains (inherited from sources)")
    if bad:
        log(f"[ln] FAIL: {len(bad)} layernorm anomalies")
        for b in bad[:10]:
            log("    " + b)
        return False
    log("[ln] PASS")
    return True


def check_tied_embedding(merged_dir: Path) -> bool:
    with (merged_dir / "config.json").open() as f:
        cfg = json.load(f)
    tied = bool(cfg.get("tie_word_embeddings", False))
    idx = open_indexed(merged_dir)
    has_embed = "model.embed_tokens.weight" in idx
    has_head = "lm_head.weight" in idx
    log(f"[tie] tie_word_embeddings={tied}, embed={has_embed}, lm_head={has_head}")
    if not has_embed:
        log("[tie] FAIL: missing model.embed_tokens.weight")
        return False
    if tied and has_head:
        e = get_tensor(idx, "model.embed_tokens.weight").float()
        h = get_tensor(idx, "lm_head.weight").float()
        if e.shape != h.shape or (e - h).abs().max().item() > 1e-3:
            log("[tie] FAIL: tie_word_embeddings=True but embed and lm_head diverge")
            return False
    if not tied and not has_head:
        log("[tie] FAIL: tie_word_embeddings=False but lm_head missing")
        return False
    log("[tie] PASS")
    return True


CRITICAL_CONFIG_FIELDS = (
    "max_position_embeddings",
    "rope_theta",
    "rope_scaling",
    "rope_parameters",
    "sliding_window",
    "use_sliding_window",
    "vocab_size",
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "model_type",
    "tie_word_embeddings",
)


def check_config_diff(merged_dir: Path, mid_dir: Path) -> bool:
    with (merged_dir / "config.json").open() as f:
        m = json.load(f)
    with (mid_dir / "config.json").open() as f:
        a = json.load(f)
    diffs = [
        (k, m.get(k, "<missing>"), a.get(k, "<missing>")) for k in CRITICAL_CONFIG_FIELDS if m.get(k) != a.get(k)
    ]
    if diffs:
        log("[cfg-diff] FAIL: critical fields differ from mid-trained:")
        for k, mv, av in diffs:
            log(f"    {k}: merged={mv!r} mid={av!r}")
        return False
    log("[cfg-diff] PASS")
    return True


def _collect_last_logits(model_dir: Path, prompts: list[str], device: str = "cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    log(f"[fwd] loading {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.bfloat16,
        device_map=device,
        low_cpu_mem_usage=True,
    )
    model.eval()
    with torch.no_grad():
        out = model(**enc)
    last_idx = enc["attention_mask"].sum(dim=1) - 1
    last_logits = out.logits[torch.arange(out.logits.shape[0]), last_idx].float().cpu()
    del model, out
    gc.collect()
    torch.cuda.empty_cache()
    return last_logits


def check_forward_and_generate(merged_dir: Path) -> bool:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = [
        "The capital of the Netherlands is",
        "In one sentence, explain why the sky is blue:",
        "Translate to Dutch: Good morning, how are you?\nDutch:",
    ]

    log("[fwd] loading model on cuda:0 (bf16)")
    tok = AutoTokenizer.from_pretrained(str(merged_dir), use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(merged_dir),
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()
    log(f"[fwd] params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    log(f"[fwd] model.config.max_position_embeddings={getattr(model.config, 'max_position_embeddings', '?')}")
    log(f"[fwd] vocab: cfg={model.config.vocab_size}, tokenizer={tok.vocab_size}")

    ok = True
    for prompt in prompts:
        enc = tok(prompt, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model(**enc)
            n_bad = int((~torch.isfinite(out.logits)).sum().item())
            topv, topi = torch.topk(out.logits[0, -1].float(), 5)
            top = [(tok.decode([int(i)]), float(v)) for i, v in zip(topi.tolist(), topv.tolist())]
            gen = model.generate(**enc, max_new_tokens=40, do_sample=False)
        log(f"[fwd] prompt={prompt!r}")
        log(f"      non_finite_logits={n_bad}   top5={top}")
        log(f"      gen={tok.decode(gen[0], skip_special_tokens=True)!r}")
        if n_bad:
            ok = False

    del model
    gc.collect()
    torch.cuda.empty_cache()
    log("[fwd] " + ("PASS" if ok else "FAIL"))
    return ok


def check_kl_vs_parents(merged_dir: Path, mid_dir: Path, base_dir: Path) -> bool:
    prompts = [
        "The capital of the Netherlands is",
        "In één zin: het weer in Amsterdam is meestal",
        "Het Koninkrijk der Nederlanden bestaat uit",
        "Python list comprehensions are useful because",
        "De stelling van Pythagoras luidt:",
    ]
    log(f"[kl] next-token distributions on {len(prompts)} prompts (sequential 3-model loads)")
    lm = _collect_last_logits(merged_dir, prompts)
    lmid = _collect_last_logits(mid_dir, prompts)
    lbase = _collect_last_logits(base_dir, prompts)

    pm = F.log_softmax(lm, dim=-1)
    pmid = F.log_softmax(lmid, dim=-1)
    pbase = F.log_softmax(lbase, dim=-1)
    e_pm = pm.exp()
    kl_mid = (e_pm * (pm - pmid)).sum(-1)
    kl_base = (e_pm * (pm - pbase)).sum(-1)
    log(
        f"[kl] KL(merged || mid)  per-prompt: {[f'{v:.3f}' for v in kl_mid.tolist()]}  mean={kl_mid.mean().item():.3f}"
    )
    log(
        f"[kl] KL(merged || base) per-prompt: {[f'{v:.3f}' for v in kl_base.tolist()]}  mean={kl_base.mean().item():.3f}"
    )
    closer = int((kl_mid < kl_base).sum().item())
    log(f"[kl] merged closer to mid on {closer}/{len(prompts)} prompts")
    if closer < int(0.6 * len(prompts)):
        log("[kl] FAIL: merged is not predominantly closer to mid")
        return False
    log("[kl] PASS")
    return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--merged", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--mid", required=True)
    p.add_argument("--base-weight", type=float, default=0.15)
    p.add_argument("--n-combo-samples", type=int, default=12)
    p.add_argument("--kl", action="store_true", help="run KL vs parents check (sequential 3-model loads)")
    p.add_argument("--skip-load", action="store_true", help="skip GPU forward/generate")
    args = p.parse_args()

    merged = Path(args.merged)
    base = Path(args.base)
    mid = Path(args.mid)

    results: list[tuple[str, bool]] = []
    hr("1/7 per-tensor finiteness")
    results.append(("finiteness", check_finiteness(merged)))
    hr("2/7 linear-combo on sampled tensors")
    results.append(("linear-combo", check_linear_combo(merged, mid, base, args.base_weight, args.n_combo_samples)))
    hr("3/7 squared-norm identity on all common tensors")
    results.append(("norm-id", check_norm_identity(merged, mid, base, args.base_weight)))
    hr("4/7 layernorm gain sanity")
    results.append(("layernorm", check_layernorm_gains(merged, mid)))
    hr("5/7 tied-embedding integrity")
    results.append(("tied-embed", check_tied_embedding(merged)))
    hr("6/7 config diff vs mid-trained")
    results.append(("cfg-diff", check_config_diff(merged, mid)))
    if not args.skip_load:
        hr("7/7 forward pass + greedy generation")
        results.append(("forward+gen", check_forward_and_generate(merged)))
    if args.kl:
        hr("8/8 KL(merged || mid) vs KL(merged || base)")
        results.append(("kl", check_kl_vs_parents(merged, mid, base)))

    hr("SUMMARY")
    width = max(len(n) for n, _ in results)
    for name, ok in results:
        log(f"  {name:<{width}}  {'PASS' if ok else 'FAIL'}")
    overall = all(ok for _, ok in results)
    log("OVERALL: " + ("PASS" if overall else "FAIL"))
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
