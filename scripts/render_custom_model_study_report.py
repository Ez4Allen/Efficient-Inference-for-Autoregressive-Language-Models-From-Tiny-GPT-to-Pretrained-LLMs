#!/usr/bin/env python3
"""Render report-ready tables and figures from a completed custom-model study."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def final_validation_metrics(path: Path) -> tuple[float | None, float | None]:
    if not path.exists():
        return None, None
    report = read_json(path)
    history = report.get("history") or []
    validation_rows = [
        row
        for row in history
        if row.get("validation_loss") is not None
        and math.isfinite(float(row["validation_loss"]))
    ]
    if not validation_rows:
        return None, None
    last = validation_rows[-1]
    loss = finite(last.get("validation_loss"))
    perplexity = finite(last.get("validation_perplexity"))
    if perplexity is None and loss is not None:
        perplexity = math.exp(min(loss, 20.0))
    return loss, perplexity


def latex_escape(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def metric(summary: dict[str, Any], name: str) -> float | None:
    return finite(summary.get(name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    study_root = args.study_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluation_path = study_root / "evaluation/custom_model_study_summary.json"
    if not evaluation_path.exists():
        raise FileNotFoundError(evaluation_path)
    evaluation = read_json(evaluation_path)
    summaries = evaluation.get("summaries")
    if not isinstance(summaries, dict) or not summaries:
        raise ValueError("Custom-model summary contains no model summaries.")

    report_paths = {
        "scratch_distill": study_root / "models/scratch_distill/training_report.json",
        "pretrain_distill": study_root / "models/pretrain_distill/training_report.json",
        "game_adapted": study_root / "models/game_adapted/training_report.json",
    }
    pretraining_report_path = study_root / "models/pretrained/pretraining_report.json"
    pretraining_loss, pretraining_perplexity = final_validation_metrics(
        pretraining_report_path
    )

    rows: list[dict[str, Any]] = []
    for model_name, payload in summaries.items():
        overall = payload.get("overall") or {}
        validation_loss, validation_perplexity = final_validation_metrics(
            report_paths.get(model_name, Path("/nonexistent"))
        )
        rows.append(
            {
                "model": model_name,
                "parameters": int(overall.get("student_parameters") or 0),
                "causal_pretraining": model_name != "scratch_distill",
                "sequence_distillation": True,
                "grounded_game_adaptation": model_name == "game_adapted",
                "pretraining_validation_loss": (
                    pretraining_loss if model_name != "scratch_distill" else None
                ),
                "pretraining_validation_perplexity": (
                    pretraining_perplexity if model_name != "scratch_distill" else None
                ),
                "final_training_validation_loss": validation_loss,
                "final_training_validation_perplexity": validation_perplexity,
                "top1_agreement": metric(overall, "top1_agreement"),
                "topk_overlap": metric(overall, "mean_topk_overlap"),
                "js_divergence": metric(overall, "mean_js_divergence"),
                "entropy_gap": metric(overall, "mean_entropy_gap"),
                "teacher_token_probability": metric(
                    overall, "mean_target_top1_probability_draft"
                ),
                "teacher_token_perplexity": metric(
                    overall, "teacher_token_perplexity_under_student"
                ),
                "rouge_l_vs_teacher": metric(overall, "rouge_l_f1"),
                "chrf_vs_teacher": metric(overall, "chrf"),
                "token_f1_vs_teacher": metric(overall, "token_f1"),
                "rouge_l_vs_formal_reference": metric(
                    overall, "reference_rouge_l_f1"
                ),
                "chrf_vs_formal_reference": metric(overall, "reference_chrf"),
                "token_f1_vs_formal_reference": metric(
                    overall, "reference_token_f1"
                ),
                "exact_speculative_acceptance": metric(
                    overall, "speculative_acceptance_rate"
                ),
                "exact_speculative_match": metric(
                    overall, "speculative_exact_match"
                ),
                "unique_conditional_top1_ratio": metric(
                    overall, "unique_draft_top1_ratio"
                ),
                "student_tokens_per_second": metric(
                    overall, "student_tokens_per_second"
                ),
            }
        )

    csv_path = output_dir / "custom_model_ablation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    diversity = evaluation.get("diversity") or {}
    diversity_rows = [
        {"model": model_name, **metrics}
        for model_name, metrics in sorted(diversity.items())
    ]
    diversity_csv = output_dir / "custom_model_diversity.csv"
    if diversity_rows:
        with diversity_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diversity_rows[0]))
            writer.writeheader()
            writer.writerows(diversity_rows)

    labels = [row["model"] for row in rows]

    def save_bar(field: str, title: str, ylabel: str, filename: str) -> None:
        values = [row.get(field) for row in rows]
        if not any(value is not None for value in values):
            return
        plotted = [float(value) if value is not None else 0.0 for value in values]
        plt.figure(figsize=(8, 5))
        plt.bar(labels, plotted)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    save_bar(
        "top1_agreement",
        "TinyQwenStudent Teacher Top-1 Agreement",
        "Agreement",
        "teacher_top1_agreement.png",
    )
    save_bar(
        "js_divergence",
        "TinyQwenStudent Distribution Divergence",
        "Jensen-Shannon divergence",
        "teacher_js_divergence.png",
    )
    save_bar(
        "exact_speculative_acceptance",
        "TinyQwenStudent Exact Speculative Acceptance",
        "Acceptance rate",
        "exact_acceptance.png",
    )
    save_bar(
        "chrf_vs_formal_reference",
        "TinyQwenStudent Held-Out chrF",
        "chrF",
        "formal_reference_chrf.png",
    )

    if diversity_rows:
        plt.figure(figsize=(8, 5))
        plt.bar(
            [row["model"] for row in diversity_rows],
            [float(row.get("distinct_2") or 0.0) for row in diversity_rows],
        )
        plt.ylabel("Distinct-2")
        plt.title("Sampled-Output Diversity (Mode-Collapse Diagnostic)")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "sampled_distinct_2.png", dpi=180)
        plt.close()

    latex_path = output_dir / "custom_model_ablation.tex"
    latex_lines = [
        r"\begin{tabular}{lrcccccc}",
        r"\toprule",
        r"Model & Params & Pretrain & Game adapt & Top-1 $\uparrow$ & JS $\downarrow$ & chrF $\uparrow$ & Accept. $\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            return "--" if value is None else f"{float(value):.3f}"

        latex_lines.append(
            f"{latex_escape(str(row['model']))} & "
            f"{int(row['parameters']):,} & "
            f"{'yes' if row['causal_pretraining'] else 'no'} & "
            f"{'yes' if row['grounded_game_adaptation'] else 'no'} & "
            f"{fmt(row['top1_agreement'])} & "
            f"{fmt(row['js_divergence'])} & "
            f"{fmt(row['chrf_vs_formal_reference'])} & "
            f"{fmt(row['exact_speculative_acceptance'])} \\\\"
        )
    latex_lines.extend([r"\bottomrule", r"\end{tabular}"])
    latex_path.write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    best_alignment = max(
        rows,
        key=lambda row: row["top1_agreement"] if row["top1_agreement"] is not None else -1,
    )
    lowest_js = min(
        rows,
        key=lambda row: row["js_divergence"] if row["js_divergence"] is not None else math.inf,
    )
    report_insert = output_dir / "CUSTOM_MODEL_REPORT_INSERT.md"
    report_insert.write_text(
        "# Custom-model result insert\n\n"
        "The team-built 43.5M-parameter architecture was evaluated under three "
        "controlled training paths with Qwen3-0.6B as the fixed teacher.  Formal "
        "evaluation prompts were held out from all optimization.\n\n"
        f"- Highest top-1 teacher agreement: `{best_alignment['model']}` "
        f"({best_alignment['top1_agreement']:.3f}).\n"
        f"- Lowest Jensen--Shannon divergence: `{lowest_js['model']}` "
        f"({lowest_js['js_divergence']:.3f}).\n"
        "- ROUGE-L/chrF/token-F1 provide standard reference-answer comparisons.\n"
        "- Distinct-n, unique-output rate, Self-BLEU, repetition, and conditional "
        "top-1 diversity are reported only as mode-collapse diagnostics.\n\n"
        "Interpret pretraining effects from `scratch_distill` versus "
        "`pretrain_distill`, and grounded adaptation effects from "
        "`pretrain_distill` versus `game_adapted`.\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "study_root": str(study_root),
        "evaluation_summary": str(evaluation_path),
        "models": labels,
        "artifacts": {
            "ablation_csv": str(csv_path),
            "diversity_csv": str(diversity_csv) if diversity_rows else None,
            "latex_table": str(latex_path),
            "report_insert": str(report_insert),
        },
        "claim_boundary": (
            "Training loss and sampled diversity do not establish serving usefulness. "
            "Teacher alignment, held-out reference quality, exact acceptance, and runtime "
            "must be interpreted together."
        ),
    }
    (output_dir / "report_artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
