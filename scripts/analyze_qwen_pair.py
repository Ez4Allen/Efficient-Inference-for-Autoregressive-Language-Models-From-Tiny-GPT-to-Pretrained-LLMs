#!/usr/bin/env python3
"""Measure draft/target token-distribution alignment on a grounded query."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.evaluation.model_pair_alignment import analyze_model_pair_on_sequence
from src.gameguide import GameGuideAssistant
from src.gameguide.prompting import build_gameguide_messages
from src.games.stardew import StardewAssistant
from src.games.terraria import TerrariaGamePlugin
from src.inference.autoregressive import greedy_decode
from src.inference.chat_runtime import QwenPairRuntime, _apply_chat_template, _eos_token_id
from src.models.runtime_config import load_qwen_pair_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--game", choices=("terraria", "stardew"), required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "gameguidelm_qwen3_pair.yaml")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plugin = TerrariaGamePlugin(auto_build=True) if args.game == "terraria" else StardewAssistant(auto_build=True)
    with GameGuideAssistant([plugin]) as assistant:
        deterministic = assistant.answer(args.question, game=args.game)
    config = load_qwen_pair_config(args.config)
    runtime = QwenPairRuntime(config)
    try:
        pair = runtime.load_pair()
        messages = build_gameguide_messages(deterministic)
        prompt_ids = _apply_chat_template(pair.target.tokenizer, messages, enable_thinking=False).to(pair.target.device)
        target_output = greedy_decode(
            pair.target.model,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_id=_eos_token_id(pair.target),
        )
        full_ids = target_output.output_ids
        draft_ids = full_ids.to(pair.draft.device)
        report = analyze_model_pair_on_sequence(
            pair.draft.model,
            pair.target.model,
            draft_ids,
            completion_start=prompt_ids.shape[1],
            top_k=args.top_k,
        )
        payload = {
            "game": args.game,
            "question": args.question,
            "prompt_tokens": prompt_ids.shape[1],
            "completion_tokens": target_output.generated_token_ids.shape[1],
            "alignment": report.to_dict(),
            "target_text": pair.target.tokenizer.decode(target_output.generated_token_ids[0], skip_special_tokens=True),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
