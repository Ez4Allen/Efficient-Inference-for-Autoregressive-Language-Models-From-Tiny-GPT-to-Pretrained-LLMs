#!/usr/bin/env python3
"""Evaluate TinyQwenStudent checkpoints against a Qwen3-0.6B teacher.

The study reports five complementary views requested by the course feedback:

1. held-out teacher alignment (top-1/top-k, JS divergence, entropy, NLL);
2. lexical/semantic reference metrics against teacher continuations;
3. language, intent, domain, and prompt-length slices;
4. sampled-output diversity and repetition diagnostics;
5. exact speculative acceptance when TinyQwenStudent drafts for Qwen3-0.6B.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.evaluation.diversity_metrics import analyze_output_diversity
from src.evaluation.model_pair_alignment import analyze_model_pair_on_sequence
from src.evaluation.reference_metrics import score_reference_answer
from src.inference.autoregressive import greedy_decode, sample_decode
from src.inference.chat_runtime import _apply_chat_template
from src.inference.speculative import greedy_speculative_decode
from src.models.loader import load_causal_lm, resolve_dtype_name
from src.models.tokenizer_contract import validate_model_tokenizer_contract
from src.utils.io import read_jsonl, write_json, write_jsonl
from src.utils.seed import set_global_seed


def parse_checkpoint(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Checkpoint must be NAME=PATH.")
    name, path = value.split("=", 1)
    if not name.strip() or not path.strip():
        raise argparse.ArgumentTypeError("Checkpoint must be NAME=PATH.")
    return name.strip(), path.strip()


def eos_ids(bundle: Any) -> int | list[int] | None:
    value = getattr(bundle.model.generation_config, "eos_token_id", None)
    return value if value is not None else bundle.tokenizer.eos_token_id


def prompt_bucket(tokens: int) -> str:
    if tokens <= 256:
        return "short_<=256"
    if tokens <= 768:
        return "medium_257_768"
    return "long_>768"


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = [
        "top1_agreement",
        "mean_topk_overlap",
        "mean_draft_entropy",
        "mean_target_entropy",
        "mean_entropy_gap",
        "mean_js_divergence",
        "target_token_logprob_draft",
        "target_token_logprob_target",
        "teacher_token_perplexity_under_student",
        "mean_target_top1_probability_draft",
        "unique_draft_top1_ratio",
        "unique_target_top1_ratio",
        "rouge_l_f1",
        "chrf",
        "token_f1",
        "reference_rouge_l_f1",
        "reference_chrf",
        "reference_token_f1",
        "speculative_acceptance_rate",
        "speculative_exact_match",
        "student_tokens_per_second",
        "prompt_tokens",
        "teacher_generated_tokens",
        "student_generated_tokens",
        "student_parameters",
    ]
    result: dict[str, Any] = {"examples": len(rows)}
    for field in numeric_fields:
        values = [
            float(row[field])
            for row in rows
            if row.get(field) is not None and not math.isnan(float(row[field]))
        ]
        result[field] = mean(values)
    return result


def grouped_summary(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "unknown")].append(row)
    return {key: aggregate(group) for key, group in sorted(groups.items())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=parse_checkpoint,
        required=True,
        help="Repeat NAME=PATH for scratch/pretrained/game-adapted students.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "held_out"),
        default="held_out",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--speculative", action="store_true")
    parser.add_argument("--draft-tokens-per-round", type=int, default=4)
    parser.add_argument("--diversity-prompts", type=int, default=12)
    parser.add_argument("--samples-per-prompt", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be positive.")
    set_global_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prompts = [
        row for row in read_jsonl(args.input) if str(row.get("split")) == args.split
    ]
    if args.limit is not None:
        prompts = prompts[: max(0, args.limit)]
    if not prompts:
        raise ValueError(f"No {args.split} prompts found.")

    teacher = load_causal_lm(
        args.teacher_model,
        tokenizer_name=args.tokenizer or args.teacher_model,
        device=args.device,
        dtype=resolve_dtype_name(args.dtype),
        local_files_only=args.local_files_only,
    )
    teacher.model.eval()
    teacher_parameters = sum(parameter.numel() for parameter in teacher.model.parameters())

    all_model_rows: dict[str, list[dict[str, Any]]] = {}
    diversity_outputs: list[dict[str, Any]] = []

    try:
        # Generate the fixed teacher reference once.  All student variants are
        # scored against exactly the same continuation.
        teacher_sequences: list[dict[str, Any]] = []
        for index, record in enumerate(prompts, start=1):
            messages = record.get("messages")
            if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
                raise ValueError(f"Prompt {record.get('id')} must end with user.")
            prompt_ids = _apply_chat_template(
                teacher.tokenizer,
                messages,
                enable_thinking=False,
            ).to(teacher.device)
            teacher_output = greedy_decode(
                teacher.model,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=eos_ids(teacher),
            )
            teacher_text = teacher.tokenizer.decode(
                teacher_output.generated_token_ids[0],
                skip_special_tokens=True,
            ).strip()
            teacher_sequences.append(
                {
                    "record": record,
                    "prompt_ids": prompt_ids,
                    "full_ids": teacher_output.output_ids,
                    "teacher_tokens": teacher_output.generated_token_ids,
                    "teacher_text": teacher_text,
                    "formal_reference": record.get("reference_answer"),
                }
            )
            print(f"Teacher [{index}/{len(prompts)}] {record['id']}")

        for model_name, checkpoint in args.checkpoint:
            print(f"\n=== Evaluating {model_name}: {checkpoint} ===")
            student = load_causal_lm(
                checkpoint,
                tokenizer_name=args.tokenizer or args.teacher_model,
                device=args.device,
                dtype=resolve_dtype_name(args.dtype),
                local_files_only=True,
            )
            student.model.eval()
            student_parameters = sum(
                parameter.numel() for parameter in student.model.parameters()
            )
            validate_model_tokenizer_contract(student.model, teacher.tokenizer)
            rows: list[dict[str, Any]] = []

            try:
                for index, sequence in enumerate(teacher_sequences, start=1):
                    record = sequence["record"]
                    prompt_ids = sequence["prompt_ids"].to(student.device)
                    full_ids = sequence["full_ids"].to(student.device)
                    completion_start = int(prompt_ids.shape[1])

                    alignment = analyze_model_pair_on_sequence(
                        student.model,
                        teacher.model,
                        full_ids,
                        completion_start=completion_start,
                        top_k=args.top_k,
                    )
                    student_output = greedy_decode(
                        student.model,
                        prompt_ids,
                        max_new_tokens=args.max_new_tokens,
                        eos_token_id=eos_ids(student),
                    )
                    student_text = student.tokenizer.decode(
                        student_output.generated_token_ids[0],
                        skip_special_tokens=True,
                    ).strip()
                    teacher_reference_metrics = score_reference_answer(
                        student_text,
                        sequence["teacher_text"],
                    )
                    formal_reference = str(
                        sequence.get("formal_reference") or ""
                    ).strip()
                    formal_reference_metrics = (
                        score_reference_answer(student_text, formal_reference)
                        if formal_reference
                        else None
                    )

                    acceptance = None
                    exact_match = None
                    if args.speculative:
                        speculative = greedy_speculative_decode(
                            student.model,
                            teacher.model,
                            prompt_ids,
                            max_new_tokens=args.max_new_tokens,
                            draft_tokens_per_round=args.draft_tokens_per_round,
                            eos_token_id=eos_ids(teacher),
                            verification_mode="exact",
                        )
                        acceptance = speculative.acceptance_rate
                        exact_match = float(
                            torch.equal(
                                speculative.generated_token_ids,
                                sequence["teacher_tokens"].to(speculative.generated_token_ids.device),
                            )
                        )

                    generated_count = int(student_output.generated_token_ids.shape[1])
                    tokens_per_second = (
                        generated_count / student_output.total_time_seconds
                        if student_output.total_time_seconds > 0
                        else 0.0
                    )
                    row = {
                        "model": model_name,
                        "checkpoint": checkpoint,
                        "id": record["id"],
                        "language": record.get("language"),
                        "domain": record.get("domain"),
                        "category": record.get("category"),
                        "intent": record.get("intent"),
                        "prompt_type": record.get("prompt_type"),
                        "expected_status": record.get("expected_status"),
                        "prompt_tokens": completion_start,
                        "prompt_bucket": prompt_bucket(completion_start),
                        "teacher_generated_tokens": int(sequence["teacher_tokens"].shape[1]),
                        "student_generated_tokens": generated_count,
                        "student_parameters": student_parameters,
                        "teacher_text": sequence["teacher_text"],
                        "student_text": student_text,
                        **alignment.to_dict(),
                        **teacher_reference_metrics.to_dict(),
                        "reference_rouge_l_f1": (
                            formal_reference_metrics.rouge_l_f1
                            if formal_reference_metrics is not None
                            else None
                        ),
                        "reference_chrf": (
                            formal_reference_metrics.chrf
                            if formal_reference_metrics is not None
                            else None
                        ),
                        "reference_token_f1": (
                            formal_reference_metrics.token_f1
                            if formal_reference_metrics is not None
                            else None
                        ),
                        "formal_reference_text": formal_reference or None,
                        "teacher_token_perplexity_under_student": float(
                            math.exp(min(-alignment.target_token_logprob_draft, 20.0))
                        ),
                        "speculative_acceptance_rate": acceptance,
                        "speculative_exact_match": exact_match,
                        "student_tokens_per_second": tokens_per_second,
                    }
                    rows.append(row)
                    print(f"{model_name} [{index}/{len(teacher_sequences)}] {record['id']}")

                    if index <= args.diversity_prompts:
                        texts: list[str] = []
                        for sample_index in range(args.samples_per_prompt):
                            generator = torch.Generator(device=student.device)
                            generator.manual_seed(
                                args.seed + index * 10_000 + sample_index
                            )
                            sampled = sample_decode(
                                student.model,
                                prompt_ids,
                                args.max_new_tokens,
                                eos_token_id=eos_ids(student),
                                temperature=args.temperature,
                                top_p=args.top_p,
                                generator=generator,
                            )
                            texts.append(
                                student.tokenizer.decode(
                                    sampled.generated_token_ids[0],
                                    skip_special_tokens=True,
                                ).strip()
                            )
                        diversity_outputs.append(
                            {
                                "model": model_name,
                                "id": record["id"],
                                "language": record.get("language"),
                                "category": record.get("category"),
                                "samples": texts,
                                "metrics": analyze_output_diversity(texts).to_dict(),
                            }
                        )
            finally:
                del student.model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_model_rows[model_name] = rows
            write_jsonl(args.output_dir / f"{model_name}_rows.jsonl", rows)

        summaries: dict[str, Any] = {}
        for model_name, rows in all_model_rows.items():
            summaries[model_name] = {
                "overall": aggregate(rows),
                "by_language": grouped_summary(rows, "language"),
                "by_domain": grouped_summary(rows, "domain"),
                "by_category": grouped_summary(rows, "category"),
                "by_intent": grouped_summary(rows, "intent"),
                "by_prompt_type": grouped_summary(rows, "prompt_type"),
                "by_prompt_bucket": grouped_summary(rows, "prompt_bucket"),
            }

        diversity_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in diversity_outputs:
            diversity_by_model[item["model"]].append(item["metrics"])
        diversity_summary = {
            model_name: {
                "prompts": len(items),
                **{
                    field: mean([float(item[field]) for item in items])
                    for field in (
                        "unique_output_rate",
                        "distinct_1",
                        "distinct_2",
                        "self_bleu_4",
                        "trigram_repetition_rate",
                        "mean_output_tokens",
                    )
                },
            }
            for model_name, items in sorted(diversity_by_model.items())
        }

        report = {
            "schema_version": 2,
            "input": str(args.input.resolve()),
            "split": args.split,
            "teacher_model": args.teacher_model,
            "teacher_tokenizer": args.tokenizer or args.teacher_model,
            "teacher_parameters": teacher_parameters,
            "models": {
                name: {
                    "checkpoint": checkpoint,
                    "parameters": int(
                        summaries[name]["overall"].get("student_parameters", 0)
                    ),
                }
                for name, checkpoint in args.checkpoint
            },
            "prompts": len(prompts),
            "max_new_tokens": args.max_new_tokens,
            "alignment_top_k": args.top_k,
            "speculative_exact_evaluation": args.speculative,
            "summaries": summaries,
            "diversity": diversity_summary,
            "metric_interpretation": {
                "teacher_alignment": (
                    "Top-1/top-k agreement, JS divergence, entropy, and teacher-token "
                    "likelihood measure whether the custom student matches Qwen3-0.6B."
                ),
                "standard_reference_metrics": (
                    "ROUGE-L, chrF, and token F1 against formal held-out reference answers "
                    "provide external text-quality baselines in addition to teacher alignment."
                ),
                "diversity": (
                    "Distinct-n, unique-output rate, Self-BLEU, repetition, and conditional "
                    "top-1 diversity are mode-collapse diagnostics, not creative-quality goals."
                ),
            },
            "claim_boundary": (
                "Sampled-output diversity is an auxiliary mode-collapse check. "
                "The primary custom-model objective is teacher alignment and "
                "speculative acceptance, not creative variability. Formal evaluation "
                "prompts are used only at evaluation time."
            ),
        }
        write_jsonl(args.output_dir / "diversity_samples.jsonl", diversity_outputs)
        write_json(args.output_dir / "custom_model_study_summary.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        del teacher.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
