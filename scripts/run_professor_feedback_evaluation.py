#!/usr/bin/env python3
"""Materialize every quantitative artifact requested after the presentation.

The command is intentionally post-processing only: it reads frozen model outputs
and never regenerates answers or changes checkpoints.  It produces standard
reference metrics, observed prompt/answer size distributions, and an optional
worked answer-validation trace for the final report.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import read_jsonl, write_jsonl


def run(command: list[str | Path]) -> None:
    rendered = [str(item) for item in command]
    print("\n$", " ".join(rendered), flush=True)
    subprocess.run(rendered, cwd=PROJECT_ROOT, check=True)



def load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        return read_jsonl(path)
    if path.suffix.casefold() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("Prediction rows must be JSONL or CSV.")


def combine_with_deterministic_baseline(
    quality_rows: Path,
    deterministic_rows: Path,
    output: Path,
) -> dict[str, Any]:
    """Add the deterministic evidence renderer as a matched baseline.

    Only IDs already present in the frozen Qwen quality rows are retained, so
    every condition is compared on the same held-out examples.
    """

    frozen = load_prediction_rows(quality_rows)
    evaluation_ids = {str(row.get("id") or "").strip() for row in frozen}
    evaluation_ids.discard("")
    deterministic = read_jsonl(deterministic_rows)
    baseline: list[dict[str, Any]] = []
    for row in deterministic:
        annotation = row.get("annotation") or {}
        result = row.get("result") or {}
        record_id = str(annotation.get("id") or row.get("id") or "").strip()
        answer = str(result.get("answer") or row.get("output") or "").strip()
        if record_id not in evaluation_ids or not answer:
            continue
        baseline.append(
            {
                "id": record_id,
                "condition": "deterministic_evidence_renderer",
                "output": answer,
            }
        )

    combined = [dict(row) for row in frozen] + baseline
    write_jsonl(output, combined)
    return {
        "frozen_quality_rows": len(frozen),
        "matched_evaluation_ids": len(evaluation_ids),
        "deterministic_baseline_rows": len(baseline),
        "combined_rows": len(combined),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality-rows", type=Path, required=True)
    parser.add_argument("--references", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deterministic-rows", type=Path)
    parser.add_argument("--trace-id")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument(
        "--bertscore-model",
        default="bert-base-multilingual-cased",
    )
    parser.add_argument("--prompt-limit", type=int)
    parser.add_argument("--answer-limit", type=int, default=192)
    parser.add_argument("--evidence-source-limit", type=int, default=6)
    parser.add_argument("--evidence-character-limit", type=int, default=14000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_rows = args.output_dir / "reference_metric_rows.jsonl"
    combined_predictions = args.output_dir / "reference_metric_predictions.jsonl"
    reference_summary = args.output_dir / "reference_metric_summary.json"
    size_summary = args.output_dir / "prompt_answer_size_summary.json"

    baseline_summary: dict[str, Any] | None = None
    reference_predictions = args.quality_rows
    if args.deterministic_rows is not None:
        baseline_summary = combine_with_deterministic_baseline(
            args.quality_rows,
            args.deterministic_rows,
            combined_predictions,
        )
        reference_predictions = combined_predictions

    reference_command: list[str | Path] = [
        sys.executable,
        "scripts/evaluate_reference_metrics.py",
        "--predictions",
        reference_predictions,
        "--references",
        *args.references,
        "--output-rows",
        reference_rows,
        "--output-summary",
        reference_summary,
        "--bertscore-model",
        args.bertscore_model,
    ]
    if args.bertscore:
        reference_command.append("--bertscore")
    run(reference_command)

    size_command: list[str | Path] = [
        sys.executable,
        "scripts/audit_prompt_answer_sizes.py",
        "--input",
        args.quality_rows,
        "--output",
        size_summary,
        "--answer-limit",
        str(args.answer_limit),
        "--evidence-source-limit",
        str(args.evidence_source_limit),
        "--evidence-character-limit",
        str(args.evidence_character_limit),
    ]
    if args.prompt_limit is not None:
        size_command.extend(["--prompt-limit", str(args.prompt_limit)])
    run(size_command)

    trace_output: Path | None = None
    if args.deterministic_rows is not None and args.trace_id:
        trace_output = args.output_dir / f"validation_trace_{args.trace_id}.json"
        run(
            [
                sys.executable,
                "scripts/explain_answer_validation.py",
                "--input",
                args.deterministic_rows,
                "--id",
                args.trace_id,
                "--output",
                trace_output,
            ]
        )

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "frozen_quality_rows": str(args.quality_rows.resolve()),
        "reference_files": [str(path.resolve()) for path in args.references],
        "artifacts": {
            "reference_metric_predictions": str(reference_predictions.resolve()),
            "reference_metric_rows": str(reference_rows.resolve()),
            "reference_metric_summary": str(reference_summary.resolve()),
            "prompt_answer_size_summary": str(size_summary.resolve()),
            "answer_validation_trace": (
                str(trace_output.resolve()) if trace_output is not None else None
            ),
        },
        "feedback_coverage": {
            "matched_deterministic_baseline": baseline_summary,
            "external_reference_benchmark": [
                "ROUGE-L",
                "chrF",
                "token F1",
                *( ["BERTScore"] if args.bertscore else [] ),
            ],
            "answer_validation_explained": trace_output is not None,
            "deterministic_evidence_renderer_in_reference_table": baseline_summary is not None,
            "observed_prompt_answer_maxima_reported": True,
            "custom_model_diversity": (
                "Produced by scripts/evaluate_custom_model_study.py after the GPU study."
            ),
        },
        "claim_boundary": (
            "These metrics post-process frozen predictions. Reference similarity "
            "complements but does not replace fact, citation, status, and numeric checks."
        ),
    }
    manifest_path = args.output_dir / "professor_feedback_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
