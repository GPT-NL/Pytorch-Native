"""HF -> OLMo-core conversion for llama3-26B / gpt-nl.

The upstream ``convert_checkpoint_from_hf`` CLI does not list ``llama3_26b`` in
its ``--model-arch`` allow-list. This wrapper calls the function directly with
the same factories used in ``scripts/train_context.py``.
"""

from __future__ import annotations

# olmo_core.kernels.moe applies @triton.autotune at import time, which fails on
# CPU-only nodes ("0 active drivers"). Forcing `import triton` to raise lets
# olmo_core.ops.moe take its ImportError branch (kernels=None). Llama does not
# invoke the MoE kernels.
import sys as _sys

_sys.modules.setdefault("triton", None)
_sys.modules.setdefault("triton.language", None)

import argparse
import json
from pathlib import Path

import torch
from examples.huggingface.convert_checkpoint_from_hf import convert_checkpoint_from_hf
from olmo_core.config import DType
from olmo_core.data.tokenizer import TokenizerConfig
from olmo_core.nn.transformer.config import TransformerConfig
from olmo_core.utils import prepare_cli_environment

DTYPE_MAP = {
    "float32": DType.float32,
    "bfloat16": DType.bfloat16,
    "float16": DType.float16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", type=Path, required=True)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("-s", "--max-sequence-length", type=int, default=32768)
    p.add_argument("--device", type=torch.device, default=None)
    p.add_argument("--dtype", choices=sorted(DTYPE_MAP), default="bfloat16")
    p.add_argument("--validate", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_config = TokenizerConfig.gpt_nl()
    transformer_config = TransformerConfig.llama3_26B(
        vocab_size=tokenizer_config.padded_vocab_size(),
        dtype=DTYPE_MAP[args.dtype],
    )
    transformer_config_dict = transformer_config.as_config_dict()
    tokenizer_config_dict = tokenizer_config.as_config_dict()

    # model_id selects MODEL_SPECIFIC_HF_TO_OLMO_CORE_MAPPINGS in
    # olmo_core.nn.hf.convert. Without it the default mapping treats HF as
    # olmo2-style double-norm Llama and feed_forward_norm is left uninitialized.
    convert_checkpoint_from_hf(
        hf_checkpoint_path=args.input,
        output_path=args.output,
        transformer_config_dict=transformer_config_dict,
        tokenizer_config_dict=tokenizer_config_dict,
        # model_id="meta-llama/Meta-Llama-3-8B-Instruct", # does not work, comment out
        # max_sequence_length=args.max_sequence_length, # does not work, comment out
        validate=args.validate,
        device=args.device,
    )

    # Upstream writes config.json via an unflushed NamedTemporaryFile and the
    # destination ends up empty. Overwrite with the same content.
    config_path = Path(args.output) / "config.json"
    with config_path.open("w") as f:
        json.dump(
            {"model": transformer_config_dict, "dataset": {"tokenizer": tokenizer_config_dict}},
            f,
            indent=2,
        )


if __name__ == "__main__":
    prepare_cli_environment()
    main()
