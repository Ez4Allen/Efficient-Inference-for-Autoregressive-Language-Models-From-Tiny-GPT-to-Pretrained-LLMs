
#!/usr/bin/env python3
"""Run the declared GameGuideLM ablation matrix over reviewed evaluation files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.experiment_matrix import load_experiment_matrix
from src.evaluation.gameguide_eval import evaluate_files
from src.gameguide import (
    EvidenceSelectionConfig,
    GameGuideAssistant,
    GameGuideQwenGenerator,
    UngroundedQwenGenerator,
)
from src.games.stardew import StardewAssistant
from src.games.terraria import TerrariaGamePlugin
from src.inference.chat_runtime import QwenPairRuntime
from src.models.runtime_config import load_qwen_pair_config
from src.utils.io import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=PROJECT_ROOT / "configs" / "gameguidelm_experiments.yaml")
    parser.add_argument("--model-config", type=Path, default=PROJECT_ROOT / "configs" / "gameguidelm_qwen3_pair.yaml")
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--default-game", choices=("terraria", "stardew_valley"))
    return parser.parse_args()


def create_generator(condition, model_config):
    if condition.generator == "deterministic":
        return None
    runtime = QwenPairRuntime(model_config)
    if condition.generator == "ungrounded":
        return UngroundedQwenGenerator(
            runtime,
            engine=condition.engine,
            fallback_on_error=model_config.grounding.fallback_on_error,
        )
    return GameGuideQwenGenerator(
        runtime,
        engine=condition.engine,
        require_citations=model_config.grounding.require_citations,
        fallback_on_error=model_config.grounding.fallback_on_error,
        max_answer_chars=model_config.grounding.max_answer_chars,
        prompt_mode=condition.prompt_mode,
        evidence_config=EvidenceSelectionConfig(
            policy=condition.evidence_policy,
            max_sources=model_config.grounding.max_evidence_sources,
            max_characters=model_config.grounding.max_evidence_characters,
        ),
        max_repair_attempts=model_config.grounding.max_repair_attempts,
    )


def main() -> None:
    args = parse_args()
    matrix = load_experiment_matrix(args.matrix)
    model_config = load_qwen_pair_config(args.model_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = {
        "matrix": matrix.to_dict(),
        "conditions": {},
    }

    for condition in matrix.conditions:
        print(f"\n=== {condition.name} ===")
        generator = create_generator(condition, model_config)
        rows_path = args.output_dir / f"{condition.name}.jsonl"
        summary_path = args.output_dir / f"{condition.name}_summary.json"
        with GameGuideAssistant(
            [TerrariaGamePlugin(auto_build=True), StardewAssistant(auto_build=True)],
            generator=generator,
        ) as assistant:
            summary = evaluate_files(
                assistant,
                args.input,
                output_path=rows_path,
                summary_path=summary_path,
                default_game=args.default_game,
            )
        combined["conditions"][condition.name] = {
            "condition": condition.to_dict(),
            "summary": summary,
            "rows_path": str(rows_path),
            "summary_path": str(summary_path),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    write_json(args.output_dir / "combined_summary.json", combined)
    print(f"\nWrote experiment outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
