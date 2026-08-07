#!/usr/bin/env python3
"""Warm repeated target/draft/speculative benchmark for GameGuideLM.

The Qwen pair is loaded once. Every engine receives explicit warm-up runs,
followed by repeated measurements over identical evidence-conditioned prompts.
Checkpoint download and model loading are therefore excluded from the reported
steady-state generation latency.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.gameguide_eval import normalize_annotation
from src.evaluation.model_benchmark import (
    basic_environment_metadata,
    first_token_mismatch,
    sha256_text,
    sha256_token_ids,
    summarize_benchmark_rows,
    token_agreement_rate,
    validate_engines,
)
from src.gameguide import EvidenceSelectionConfig, GameGuideAssistant
from src.gameguide.prompting import prepare_gameguide_prompt
from src.gameguide.validation import validate_gameguide_answer
from src.games.stardew import StardewAssistant
from src.games.terraria import TerrariaGamePlugin
from src.inference.chat_runtime import QwenPairRuntime
from src.models.runtime_config import load_qwen_pair_config
from src.utils.io import read_jsonl, write_json, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark warm GameGuideLM target, draft, and speculative "
            "generation in one process."
        )
    )
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "gameguidelm_qwen3_pair.yaml",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        default=["target", "draft", "speculative"],
    )
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--draft-tokens-per-round", type=int)
    parser.add_argument(
        "--verification-mode",
        choices=("exact", "block"),
        help=(
            "Override generation.verification_mode. Use exact for target-token "
            "identity checks and block for wall-clock speed experiments."
        ),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--default-game",
        choices=("terraria", "stardew_valley"),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("evidence_only", "scaffolded"),
        default="evidence_only",
    )
    parser.add_argument(
        "--evidence-policy",
        choices=("compact", "full", "structured_only", "guide_only"),
        default="compact",
    )
    parser.add_argument("--max-evidence-sources", type=int, default=6)
    parser.add_argument("--max-evidence-characters", type=int, default=14_000)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    args.engines = validate_engines(args.engines)
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive when provided.")
    if args.draft_tokens_per_round is not None and args.draft_tokens_per_round <= 0:
        raise ValueError("--draft-tokens-per-round must be positive when provided.")
    return args


def _load_annotations(
    paths: Iterable[Path],
    *,
    default_game: str | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for record in read_jsonl(path):
            records.append(
                normalize_annotation(
                    record,
                    source_path=path,
                    default_game=default_game,
                )
            )
            if limit is not None and len(records) >= limit:
                return records
    return records


def _prepare_cases(
    annotations: list[dict[str, Any]],
    *,
    prompt_mode: str,
    evidence_config: EvidenceSelectionConfig,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with GameGuideAssistant(
        [TerrariaGamePlugin(auto_build=True), StardewAssistant(auto_build=True)]
    ) as assistant:
        for index, annotation in enumerate(annotations):
            result = assistant.answer(
                annotation["question"],
                game=annotation["game"],
                player_state=annotation.get("player_state"),
                include_debug=False,
            )
            if result.status != "found":
                continue
            prepared = prepare_gameguide_prompt(
                result,
                prompt_mode=prompt_mode,
                evidence_config=evidence_config,
            )
            cases.append(
                {
                    "example_id": str(annotation.get("id") or f"case_{index:05d}"),
                    "annotation": annotation,
                    "result": prepared.result,
                    "messages": prepared.messages,
                    "evidence_selection": prepared.evidence_report.to_dict(),
                }
            )
    return cases


def _run_generation(
    runtime: QwenPairRuntime,
    case: dict[str, Any],
    *,
    engine: str,
    run_index: int,
    warmup: bool,
    max_new_tokens: int,
    draft_tokens_per_round: int | None,
    verification_mode: str | None,
    require_citations: bool,
    max_answer_chars: int,
    target_text: str | None,
    target_token_ids: tuple[int, ...] | None,
) -> tuple[dict[str, Any], str, tuple[int, ...]]:
    generated = runtime.generate(
        case["messages"],
        engine=engine,
        max_new_tokens=max_new_tokens,
        draft_tokens_per_round=draft_tokens_per_round,
        verification_mode=verification_mode,
    )
    validation = validate_gameguide_answer(
        generated.text,
        case["result"],
        require_citations=require_citations,
        max_answer_chars=max_answer_chars,
    )
    compare_with_target = engine == "speculative" and target_token_ids is not None
    exact_target_match = (
        generated.generated_token_ids == target_token_ids
        if compare_with_target
        else None
    )
    exact_target_text_match = (
        generated.text == target_text
        if engine == "speculative" and target_text is not None
        else None
    )
    mismatch_index = (
        first_token_mismatch(
            target_token_ids or [],
            generated.generated_token_ids,
        )
        if compare_with_target
        else None
    )
    agreement_rate = (
        token_agreement_rate(
            target_token_ids or [],
            generated.generated_token_ids,
        )
        if compare_with_target
        else None
    )
    row = {
        "example_id": case["example_id"],
        "game": case["annotation"]["game"],
        "language": case["annotation"].get("language"),
        "intent": case["result"].intent,
        "question": case["annotation"]["question"],
        "engine": engine,
        "run": run_index,
        "warmup": warmup,
        "output_sha256": sha256_text(generated.text),
        "token_ids_sha256": sha256_token_ids(generated.generated_token_ids),
        "grounding_valid": validation.valid,
        "validation_issues": validation.issues,
        "exact_target_match": exact_target_match,
        "exact_target_text_match": exact_target_text_match,
        "first_target_token_mismatch": mismatch_index,
        "token_agreement_rate": agreement_rate,
        "evidence_selection": case["evidence_selection"],
        **generated.to_dict(),
    }
    return row, generated.text, generated.generated_token_ids


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_qwen_pair_config(args.config)
    evidence_config = EvidenceSelectionConfig(
        policy=args.evidence_policy,
        max_sources=args.max_evidence_sources,
        max_characters=args.max_evidence_characters,
    )
    annotations = _load_annotations(
        args.input,
        default_game=args.default_game,
        limit=args.limit,
    )
    cases = _prepare_cases(
        annotations,
        prompt_mode=args.prompt_mode,
        evidence_config=evidence_config,
    )
    if not cases:
        raise RuntimeError(
            "No found examples were available for model benchmarking. "
            "Build the required knowledge/guide stores or provide different inputs."
        )

    runtime = QwenPairRuntime(config)
    rows: list[dict[str, Any]] = []
    try:
        # Warm each selected engine using the same first prompt. Warm-up rows are
        # stored for auditability and excluded by summarize_benchmark_rows().
        warm_case = cases[0]
        warm_target_text: str | None = None
        warm_target_token_ids: tuple[int, ...] | None = None
        for engine in args.engines:
            for warmup_index in range(args.warmup_runs):
                row, text, token_ids = _run_generation(
                    runtime,
                    warm_case,
                    engine=engine,
                    run_index=warmup_index + 1,
                    warmup=True,
                    max_new_tokens=args.max_new_tokens,
                    draft_tokens_per_round=args.draft_tokens_per_round,
                    verification_mode=args.verification_mode,
                    require_citations=config.grounding.require_citations,
                    max_answer_chars=config.grounding.max_answer_chars,
                    target_text=warm_target_text,
                    target_token_ids=warm_target_token_ids,
                )
                rows.append(row)
                if engine == "target":
                    warm_target_text = text
                    warm_target_token_ids = token_ids

        for case in cases:
            target_text: str | None = None
            target_token_ids: tuple[int, ...] | None = None
            # validate_engines() requires target whenever speculative is present,
            # so the first measured target output is the exact-match reference.
            for engine in args.engines:
                for run_index in range(args.runs):
                    row, text, token_ids = _run_generation(
                        runtime,
                        case,
                        engine=engine,
                        run_index=run_index + 1,
                        warmup=False,
                        max_new_tokens=args.max_new_tokens,
                        draft_tokens_per_round=args.draft_tokens_per_round,
                        verification_mode=args.verification_mode,
                        require_citations=config.grounding.require_citations,
                        max_answer_chars=config.grounding.max_answer_chars,
                        target_text=target_text,
                        target_token_ids=target_token_ids,
                    )
                    rows.append(row)
                    if engine == "target" and target_token_ids is None:
                        target_text = text
                        target_token_ids = token_ids
    finally:
        runtime.close()

    summary = summarize_benchmark_rows(rows)
    summary.update(
        {
            "status": "passed",
            "configuration": {
                "model_config": str(args.config),
                "engines": list(args.engines),
                "warmup_runs": args.warmup_runs,
                "runs": args.runs,
                "max_new_tokens": args.max_new_tokens,
                "draft_tokens_per_round": (
                    args.draft_tokens_per_round
                    or config.generation.draft_tokens_per_round
                ),
                "verification_mode": (
                    args.verification_mode
                    or config.generation.verification_mode
                ),
                "prompt_mode": args.prompt_mode,
                "evidence_policy": args.evidence_policy,
                "max_evidence_sources": args.max_evidence_sources,
                "max_evidence_characters": args.max_evidence_characters,
            },
            "environment": basic_environment_metadata(),
            "input_examples": len(annotations),
            "benchmark_cases": len(cases),
        }
    )
    write_jsonl(args.output, rows)
    write_json(args.summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
