#!/usr/bin/env python3
"""Validate the configured Qwen pair and compare generation engines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.chat_runtime import QwenPairRuntime
from src.models.loader import print_model_info, validate_tokenizer_compatibility
from src.models.runtime_config import load_qwen_pair_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "terraria_qwen3_pair.yaml",
    )
    parser.add_argument(
        "--prompt",
        default="Give one concise sentence explaining what speculative decoding does.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=("draft", "target", "speculative"),
        default=("draft", "target"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_qwen_pair_config(args.config)
    runtime = QwenPairRuntime(config)
    try:
        pair = runtime.load_pair()
        validate_tokenizer_compatibility(pair.draft, pair.target)
        print_model_info(pair.draft)
        print_model_info(pair.target)
        print("\nTokenizer compatibility: passed")

        messages = [{"role": "user", "content": args.prompt}]
        for engine in args.engines:
            result = runtime.generate(
                messages,
                engine=engine,
                max_new_tokens=args.max_new_tokens,
            )
            print(f"\n=== {engine.upper()} ===")
            print(result.text)
            print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
