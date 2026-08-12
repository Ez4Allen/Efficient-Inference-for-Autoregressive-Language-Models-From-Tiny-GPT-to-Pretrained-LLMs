#!/usr/bin/env python3
"""Evaluate generated answers with standard reference-based text metrics.

This complements GameGuideLM's task-specific required-fact coverage and pass
criterion with ROUGE-L, chrF, token F1, and optional multilingual BERTScore.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reference_metrics import score_reference_answer
from src.utils.io import read_jsonl, write_json, write_jsonl


def _load_prediction_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.casefold()
    if suffix == ".jsonl":
        return read_jsonl(path)
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    raise ValueError("Predictions must be JSONL or CSV.")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _optional_bertscore(
    predictions: list[str],
    references: list[str],
    *,
    model_type: str,
    batch_size: int,
    device: str | None,
) -> list[float]:
    try:
        from bert_score import score
    except ImportError as error:
        raise RuntimeError(
            "BERTScore was requested but bert-score is not installed. "
            "Install the optional metrics dependency with "
            "`pip install 'gameguidelm[metrics]'` or `pip install bert-score`."
        ) from error

    _, _, f1 = score(
        predictions,
        references,
        model_type=model_type,
        batch_size=batch_size,
        device=device,
        verbose=True,
        lang=None,
        rescale_with_baseline=False,
    )
    return [float(value) for value in f1.cpu().tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--references", type=Path, nargs="+", required=True)
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-summary", type=Path, required=True)
    parser.add_argument("--id-field", default="id")
    parser.add_argument("--prediction-field", default="output")
    parser.add_argument("--condition-field", default="condition")
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument(
        "--bertscore-model",
        default="bert-base-multilingual-cased",
    )
    parser.add_argument("--bertscore-batch-size", type=int, default=16)
    parser.add_argument("--device")
    args = parser.parse_args()

    reference_by_id: dict[str, dict[str, Any]] = {}
    for path in args.references:
        for record in read_jsonl(path):
            record_id = str(record.get("id", "")).strip()
            if not record_id:
                raise ValueError(f"Reference record without id: {path}")
            if record_id in reference_by_id:
                raise ValueError(f"Duplicate reference id: {record_id}")
            reference_by_id[record_id] = record

    predictions = _load_prediction_rows(args.predictions)
    prepared: list[tuple[dict[str, Any], dict[str, Any], str, str]] = []
    missing_ids: list[str] = []
    for row in predictions:
        record_id = str(row.get(args.id_field, "")).strip()
        if record_id not in reference_by_id:
            missing_ids.append(record_id)
            continue
        prediction = str(row.get(args.prediction_field, "")).strip()
        reference_record = reference_by_id[record_id]
        reference = str(reference_record.get("reference_answer", "")).strip()
        if not reference:
            raise ValueError(f"Reference answer missing for {record_id}")
        prepared.append((row, reference_record, prediction, reference))

    if missing_ids:
        raise ValueError(
            "Prediction IDs missing from references: " + ", ".join(missing_ids[:20])
        )
    if not prepared:
        raise ValueError("No prediction/reference pairs were loaded.")

    bertscore_values: list[float | None] = [None] * len(prepared)
    if args.bertscore:
        bertscore_values = _optional_bertscore(
            [item[2] for item in prepared],
            [item[3] for item in prepared],
            model_type=args.bertscore_model,
            batch_size=args.bertscore_batch_size,
            device=args.device,
        )

    output_rows: list[dict[str, Any]] = []
    for (prediction_row, reference_row, prediction, reference), bertscore_f1 in zip(
        prepared,
        bertscore_values,
        strict=True,
    ):
        metrics = score_reference_answer(
            prediction,
            reference,
            bertscore_f1=bertscore_f1,
        )
        output_rows.append(
            {
                "id": str(prediction_row[args.id_field]),
                "condition": str(
                    prediction_row.get(args.condition_field) or "unspecified"
                ),
                "game": str(reference_row.get("game") or "unknown"),
                "language": str(reference_row.get("language") or "unknown"),
                "intent": str(reference_row.get("intent") or "unknown"),
                "category": str(reference_row.get("category") or "unknown"),
                "prediction": prediction,
                "reference_answer": reference,
                **metrics.to_dict(),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        grouped[row["condition"]].append(row)
        by_language[f"{row['condition']}::{row['language']}"] .append(row)
        by_intent[f"{row['condition']}::{row['intent']}"] .append(row)

    metric_names = ["rouge_l_f1", "chrf", "token_f1"]
    if args.bertscore:
        metric_names.append("bertscore_f1")

    def summarize(groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, rows in sorted(groups.items()):
            result[key] = {
                "examples": len(rows),
                **{
                    name: _mean(
                        [
                            float(row[name])
                            for row in rows
                            if row.get(name) is not None
                            and not math.isnan(float(row[name]))
                        ]
                    )
                    for name in metric_names
                },
            }
        return result

    summary = {
        "predictions": str(args.predictions.resolve()),
        "reference_files": [str(path.resolve()) for path in args.references],
        "examples": len(output_rows),
        "metrics": metric_names,
        "conditions": summarize(grouped),
        "by_language": summarize(by_language),
        "by_intent": summarize(by_intent),
        "interpretation": (
            "ROUGE-L, chrF, token F1, and optional BERTScore measure similarity "
            "to reference answers. They complement rather than replace the "
            "required-fact, forbidden-error, citation, and numeric-support checks."
        ),
    }

    write_jsonl(args.output_rows, output_rows)
    write_json(args.output_summary, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
