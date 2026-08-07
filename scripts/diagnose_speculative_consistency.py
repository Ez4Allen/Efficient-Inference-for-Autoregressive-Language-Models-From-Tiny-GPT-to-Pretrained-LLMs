#!/usr/bin/env python3
"""Diagnose q_len=1 versus block target consistency on a real checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.autoregressive import greedy_decode
from src.inference.chat_runtime import QwenPairRuntime, _apply_chat_template, _eos_token_id
from src.inference.consistency import diagnose_target_block_consistency
from src.models.runtime_config import load_qwen_pair_config
from src.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a target model's incremental greedy tokens with its "
            "q_len>1 block-verification argmax sequence."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gameguidelm_qwen3_pair.yaml",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if any(value <= 0 for value in args.block_sizes):
        raise ValueError("Every --block-sizes value must be positive.")

    config = load_qwen_pair_config(args.config)
    runtime = QwenPairRuntime(config)
    try:
        target = runtime.load_target()
        messages = [{"role": "user", "content": args.prompt}]
        input_ids = _apply_chat_template(
            target.tokenizer,
            messages,
            enable_thinking=args.enable_thinking,
        ).to(target.device)
        reference = greedy_decode(
            target.model,
            input_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=_eos_token_id(target),
        )

        reports = [
            diagnose_target_block_consistency(
                target.model,
                input_ids,
                reference.generated_token_ids,
                block_size=block_size,
            ).to_dict()
            for block_size in args.block_sizes
        ]
        payload = {
            "status": "passed",
            "model": target.model_name,
            "prompt_tokens": int(input_ids.shape[1]),
            "generated_tokens": int(reference.generated_token_ids.shape[1]),
            "reports": reports,
        }
        if args.output is not None:
            write_json(args.output, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
