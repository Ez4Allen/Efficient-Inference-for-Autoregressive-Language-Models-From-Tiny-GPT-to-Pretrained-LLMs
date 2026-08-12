#!/usr/bin/env python3
"""Orchestrate the professor-feedback custom TinyQwenStudent study.

The pipeline is resumable and creates three controlled variants:

A. scratch_distill: random initialization -> Qwen3-0.6B distillation;
B. pretrain_distill: project-local causal pretraining -> distillation;
C. game_adapted: pretrain_distill -> grounded GameGuide adaptation.

Run individual stages or the full pipeline.  Expensive model generation/training
is never hidden inside tests or import side effects.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import read_jsonl, read_yaml, write_json, write_jsonl, write_yaml
from src.utils.paths import resolve_project_path

STAGES = (
    "corpus",
    "prompts",
    "teacher",
    "pretrain",
    "scratch_distill",
    "pretrain_distill",
    "grounded_teacher",
    "game_adapt",
    "evaluate",
    "render",
)


def mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"Missing config section: {name}")
    return value


def run(command: list[str | Path]) -> None:
    rendered = [str(value) for value in command]
    print("\n$", " ".join(rendered), flush=True)
    subprocess.run(rendered, cwd=PROJECT_ROOT, check=True)


def should_run(path: Path, *, force: bool) -> bool:
    return force or not path.exists()


def write_pair_config(
    path: Path,
    *,
    teacher: Mapping[str, Any],
    game: Mapping[str, Any],
) -> None:
    model = str(teacher["model_name_or_path"])
    tokenizer = str(teacher.get("tokenizer_name_or_path") or model)
    payload = {
        "models": {
            "draft": {
                "model_name_or_path": model,
                "tokenizer_name_or_path": tokenizer,
                "adapter_path": None,
                "trust_remote_code": False,
                "local_files_only": bool(teacher.get("local_files_only", False)),
                "load_in_4bit": False,
            },
            "target": {
                "model_name_or_path": model,
                "tokenizer_name_or_path": tokenizer,
                "adapter_path": None,
                "trust_remote_code": False,
                "local_files_only": bool(teacher.get("local_files_only", False)),
                "load_in_4bit": False,
            },
        },
        "runtime": {
            "device": str(teacher.get("device", "cuda")),
            "dtype": str(teacher.get("dtype", "auto")),
        },
        "generation": {
            "engine": "target",
            "max_new_tokens": int(teacher.get("max_new_tokens", 96)),
            "draft_tokens_per_round": 4,
            "verification_mode": "exact",
            "enable_thinking": False,
        },
        "grounding": {
            "require_citations": True,
            "fallback_on_error": False,
            "max_answer_chars": 3500,
            "prompt_mode": "evidence_only",
            "evidence_policy": "compact",
            "max_evidence_sources": int(game.get("evidence_sources", 4)),
            "max_evidence_characters": int(game.get("evidence_characters", 8000)),
            "max_repair_attempts": 1,
        },
    }
    write_yaml(path, payload)


def training_config(
    *,
    path: Path,
    teacher: Mapping[str, Any],
    student: Mapping[str, Any],
    stage: Mapping[str, Any],
    stage_name: str,
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    initial_checkpoint: Path | None,
    seed: int,
) -> None:
    model = {
        "target_model_name_or_path": str(teacher["model_name_or_path"]),
        "tokenizer_name_or_path": str(
            teacher.get("tokenizer_name_or_path") or teacher["model_name_or_path"]
        ),
        "local_files_only": bool(teacher.get("local_files_only", False)),
        **dict(student),
    }
    if initial_checkpoint is not None:
        model["initial_checkpoint"] = str(initial_checkpoint)
    payload = {
        "model": model,
        "data": {
            "train_path": str(train_path),
            "validation_path": str(validation_path),
            "max_length": int(stage.get("max_length", 1024)),
            "truncation_mode": str(stage.get("truncation_mode", "preserve_assistant")),
        },
        "training": {
            "output_dir": str(output_dir),
            "stage_name": stage_name,
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_accumulation_steps": int(stage.get("gradient_accumulation_steps", 8)),
            "learning_rate": float(stage.get("learning_rate", 3.0e-4)),
            "weight_decay": float(stage.get("weight_decay", 0.01)),
            "max_steps": int(stage.get("max_steps", 1000)),
            "warmup_ratio": 0.03,
            "max_grad_norm": 1.0,
            "logging_steps": int(stage.get("logging_steps", 25)),
            "eval_steps": int(stage.get("eval_steps", 100)),
            "eval_batches": int(stage.get("eval_batches", 20)),
            "save_steps": int(stage.get("save_steps", 250)),
            "seed": seed,
            "device": str(teacher.get("device", "cuda")),
            "compute_dtype": str(teacher.get("dtype", "auto")),
            "num_workers": 0,
        },
    }
    write_yaml(path, payload)


def merge_teacher_files(paths: list[Path], output: Path, split: str) -> None:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            row = dict(row)
            row["split"] = split
            record_id = str(row.get("id"))
            if record_id in seen:
                row["id"] = f"{record_id}_{len(records):06d}"
            seen.add(str(row["id"]))
            records.append(row)
    if not records:
        raise RuntimeError(f"No grounded {split} teacher records were generated.")
    write_jsonl(output, records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/custom_model_study.yaml",
    )
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = read_yaml(resolve_project_path(args.config))
    if not isinstance(payload, Mapping):
        raise TypeError("Custom-model study config must be a mapping.")
    study = mapping(payload, "study")
    teacher = mapping(payload, "teacher")
    student = mapping(payload, "student")
    corpus_cfg = mapping(payload, "corpus")
    pretraining = mapping(payload, "pretraining")
    distillation = mapping(payload, "sequence_distillation")
    game = mapping(payload, "game_adaptation")
    prompt_cfg = mapping(payload, "prompt_pool")
    evaluation = mapping(payload, "evaluation")

    output_root = resolve_project_path(str(study.get("output_dir", "results/custom_model_study")))
    output_root.mkdir(parents=True, exist_ok=True)
    seed = int(study.get("seed", 42))
    stages = STAGES if args.stage == "all" else (args.stage,)

    corpus_path = output_root / "data/pretraining_corpus.jsonl"
    corpus_manifest = output_root / "data/pretraining_manifest.json"
    prompt_pool = output_root / "data/prompt_pool.jsonl"
    prompt_manifest = output_root / "data/prompt_pool_manifest.json"
    teacher_train = output_root / "data/teacher_train.jsonl"
    teacher_validation = output_root / "data/teacher_validation.jsonl"
    teacher_train_manifest = output_root / "data/teacher_train_manifest.json"
    teacher_validation_manifest = output_root / "data/teacher_validation_manifest.json"
    pretrain_root = output_root / "models/pretrained"
    scratch_root = output_root / "models/scratch_distill"
    pretrain_distill_root = output_root / "models/pretrain_distill"
    game_root = output_root / "models/game_adapted"

    if "corpus" in stages and should_run(corpus_path, force=args.force):
        command: list[str | Path] = [
            sys.executable,
            "scripts/build_tiny_student_corpus.py",
            "--output",
            corpus_path,
            "--manifest",
            corpus_manifest,
            "--validation-fraction",
            str(corpus_cfg.get("validation_fraction", 0.05)),
            "--seed",
            str(seed),
        ]
        if not bool(corpus_cfg.get("exclude_eval_entities", True)):
            command.append("--include-eval-entities")
        run(command)

    if "prompts" in stages and should_run(prompt_pool, force=args.force):
        command = [
            sys.executable,
            "scripts/build_student_prompt_pool.py",
            "--train-input",
            "data/stardew/sft/train.jsonl",
            "data/terraria/terraria_train_v1.jsonl",
            "--validation-input",
            "data/stardew/sft/validation.jsonl",
            "data/terraria/terraria_validation_v1.jsonl",
            "--eval-input",
            "data/stardew/evaluation/stardew_eval_v1.jsonl",
            "data/terraria/terraria_eval.jsonl",
            "--output",
            prompt_pool,
            "--manifest",
            prompt_manifest,
        ]
        if prompt_cfg.get("max_per_category") is not None:
            command.extend(["--max-per-category", str(prompt_cfg["max_per_category"])])
        if bool(prompt_cfg.get("augment_stardew_zh", True)):
            command.append("--augment-stardew-zh")
            command.extend(
                [
                    "--augmentation-validation-fraction",
                    str(prompt_cfg.get("augmentation_validation_fraction", 0.1)),
                    "--seed",
                    str(seed),
                ]
            )
            if prompt_cfg.get("augmentation_limit") is not None:
                command.extend(
                    ["--augmentation-limit", str(prompt_cfg["augmentation_limit"])]
                )
        run(command)

    if "teacher" in stages:
        for split, output, manifest in (
            ("train", teacher_train, teacher_train_manifest),
            ("validation", teacher_validation, teacher_validation_manifest),
        ):
            if should_run(output, force=args.force):
                command = [
                    sys.executable,
                    "scripts/generate_qwen_teacher_continuations.py",
                    "--input",
                    prompt_pool,
                    "--output",
                    output,
                    "--manifest",
                    manifest,
                    "--model",
                    str(teacher["model_name_or_path"]),
                    "--tokenizer",
                    str(teacher.get("tokenizer_name_or_path") or teacher["model_name_or_path"]),
                    "--split",
                    split,
                    "--max-new-tokens",
                    str(teacher.get("max_new_tokens", 96)),
                    "--device",
                    str(teacher.get("device", "cuda")),
                    "--dtype",
                    str(teacher.get("dtype", "auto")),
                ]
                if bool(teacher.get("local_files_only", False)):
                    command.append("--local-files-only")
                if args.force:
                    command.append("--force")
                run(command)

    pretraining_config = output_root / "configs/pretraining.yaml"
    if "pretrain" in stages and should_run(pretrain_root / "final/config.json", force=args.force):
        write_yaml(
            pretraining_config,
            {
                "model": {
                    "teacher_model_name_or_path": str(teacher["model_name_or_path"]),
                    "tokenizer_name_or_path": str(
                        teacher.get("tokenizer_name_or_path") or teacher["model_name_or_path"]
                    ),
                    "local_files_only": bool(teacher.get("local_files_only", False)),
                    **dict(student),
                },
                "data": {
                    "corpus_path": str(corpus_path),
                    "max_length": int(pretraining.get("max_length", 512)),
                    "stride": int(pretraining.get("stride", 512)),
                },
                "training": {
                    "output_dir": str(pretrain_root),
                    "per_device_train_batch_size": 1,
                    "per_device_eval_batch_size": 1,
                    "gradient_accumulation_steps": int(pretraining.get("gradient_accumulation_steps", 16)),
                    "learning_rate": float(pretraining.get("learning_rate", 3.0e-4)),
                    "weight_decay": float(pretraining.get("weight_decay", 0.01)),
                    "max_steps": int(pretraining.get("max_steps", 1000)),
                    "warmup_ratio": 0.03,
                    "max_grad_norm": 1.0,
                    "logging_steps": int(pretraining.get("logging_steps", 25)),
                    "eval_steps": int(pretraining.get("eval_steps", 100)),
                    "eval_batches": int(pretraining.get("eval_batches", 20)),
                    "save_steps": int(pretraining.get("save_steps", 250)),
                    "seed": seed,
                    "device": str(teacher.get("device", "cuda")),
                    "compute_dtype": str(teacher.get("dtype", "auto")),
                    "num_workers": 0,
                },
            },
        )
        run([sys.executable, "scripts/pretrain_tiny_qwen_student.py", "--config", pretraining_config])

    scratch_config = output_root / "configs/scratch_distill.yaml"
    if "scratch_distill" in stages and should_run(scratch_root / "final/config.json", force=args.force):
        training_config(
            path=scratch_config,
            teacher=teacher,
            student=student,
            stage=distillation,
            stage_name="qwen3_0_6b_sequence_distillation_from_scratch",
            train_path=teacher_train,
            validation_path=teacher_validation,
            output_dir=scratch_root,
            initial_checkpoint=None,
            seed=seed,
        )
        run([sys.executable, "scripts/train_tiny_qwen_draft.py", "--config", scratch_config])

    pretrain_distill_config = output_root / "configs/pretrain_distill.yaml"
    if "pretrain_distill" in stages and should_run(pretrain_distill_root / "final/config.json", force=args.force):
        training_config(
            path=pretrain_distill_config,
            teacher=teacher,
            student=student,
            stage=distillation,
            stage_name="qwen3_0_6b_sequence_distillation_after_pretraining",
            train_path=teacher_train,
            validation_path=teacher_validation,
            output_dir=pretrain_distill_root,
            initial_checkpoint=pretrain_root / "final",
            seed=seed,
        )
        run([sys.executable, "scripts/train_tiny_qwen_draft.py", "--config", pretrain_distill_config])

    grounded_pair_config = output_root / "configs/qwen0_6b_grounded_teacher.yaml"
    grounded_train = output_root / "data/grounded_teacher_train.jsonl"
    grounded_validation = output_root / "data/grounded_teacher_validation.jsonl"
    if "grounded_teacher" in stages and should_run(grounded_train, force=args.force):
        write_pair_config(grounded_pair_config, teacher=teacher, game=game)
        temporary: dict[tuple[str, str], Path] = {}
        sources = {
            ("train", "stardew_valley"): Path("data/stardew/sft/train.jsonl"),
            ("validation", "stardew_valley"): Path("data/stardew/sft/validation.jsonl"),
            ("train", "terraria"): Path("data/terraria/terraria_train_v1.jsonl"),
            ("validation", "terraria"): Path("data/terraria/terraria_validation_v1.jsonl"),
        }
        for (split, game_name), source in sources.items():
            output = output_root / f"data/grounded_{split}_{game_name}.jsonl"
            temporary[(split, game_name)] = output
            if should_run(output, force=args.force):
                command = [
                    sys.executable,
                    "scripts/generate_teacher_answers.py",
                    "--input",
                    source,
                    "--output",
                    output,
                    "--config",
                    grounded_pair_config,
                    "--default-game",
                    game_name,
                    "--split",
                    split,
                    "--target-source",
                    "validated_qwen3_0_6b_grounded_teacher",
                ]
                run(command)
        merge_teacher_files(
            [temporary[("train", "stardew_valley")], temporary[("train", "terraria")]],
            grounded_train,
            "train",
        )
        merge_teacher_files(
            [
                temporary[("validation", "stardew_valley")],
                temporary[("validation", "terraria")],
            ],
            grounded_validation,
            "validation",
        )

    game_config = output_root / "configs/game_adapt.yaml"
    if "game_adapt" in stages and should_run(game_root / "final/config.json", force=args.force):
        training_config(
            path=game_config,
            teacher=teacher,
            student=student,
            stage=game,
            stage_name="grounded_gameguide_adaptation_after_pretraining_and_distillation",
            train_path=grounded_train,
            validation_path=grounded_validation,
            output_dir=game_root,
            initial_checkpoint=pretrain_distill_root / "final",
            seed=seed,
        )
        run([sys.executable, "scripts/train_tiny_qwen_draft.py", "--config", game_config])

    evaluation_summary = output_root / "evaluation/custom_model_study_summary.json"
    if "evaluate" in stages and should_run(evaluation_summary, force=args.force):
        command = [
            sys.executable,
            "scripts/evaluate_custom_model_study.py",
            "--input",
            prompt_pool,
            "--teacher-model",
            str(teacher["model_name_or_path"]),
            "--tokenizer",
            str(teacher.get("tokenizer_name_or_path") or teacher["model_name_or_path"]),
            "--checkpoint",
            f"scratch_distill={scratch_root / 'final'}",
            "--checkpoint",
            f"pretrain_distill={pretrain_distill_root / 'final'}",
            "--checkpoint",
            f"game_adapted={game_root / 'final'}",
            "--output-dir",
            output_root / "evaluation",
            "--split",
            str(evaluation.get("split", "validation")),
            "--max-new-tokens",
            str(evaluation.get("max_new_tokens", 64)),
            "--top-k",
            str(evaluation.get("top_k", 5)),
            "--device",
            str(teacher.get("device", "cuda")),
            "--dtype",
            str(teacher.get("dtype", "auto")),
            "--draft-tokens-per-round",
            str(evaluation.get("draft_tokens_per_round", 4)),
            "--diversity-prompts",
            str(evaluation.get("diversity_prompts", 12)),
            "--samples-per-prompt",
            str(evaluation.get("samples_per_prompt", 5)),
            "--temperature",
            str(evaluation.get("temperature", 0.8)),
            "--top-p",
            str(evaluation.get("top_p", 0.9)),
            "--seed",
            str(seed),
        ]
        if evaluation.get("limit") is not None:
            command.extend(["--limit", str(evaluation["limit"])])
        if bool(evaluation.get("speculative", True)):
            command.append("--speculative")
        if bool(teacher.get("local_files_only", False)):
            command.append("--local-files-only")
        run(command)

    report_manifest = output_root / "report/report_artifact_manifest.json"
    if "render" in stages and should_run(report_manifest, force=args.force):
        if not evaluation_summary.exists():
            raise FileNotFoundError(
                "Custom-model evaluation is required before report rendering: "
                f"{evaluation_summary}"
            )
        run(
            [
                sys.executable,
                "scripts/render_custom_model_study_report.py",
                "--study-root",
                output_root,
                "--output-dir",
                output_root / "report",
            ]
        )

    manifest = {
        "schema_version": 1,
        "config": str(resolve_project_path(args.config)),
        "output_root": str(output_root),
        "stage": args.stage,
        "artifacts": {
            "corpus": str(corpus_path),
            "prompt_pool": str(prompt_pool),
            "teacher_train": str(teacher_train),
            "teacher_validation": str(teacher_validation),
            "pretrained_checkpoint": str(pretrain_root / "final"),
            "scratch_distill_checkpoint": str(scratch_root / "final"),
            "pretrain_distill_checkpoint": str(pretrain_distill_root / "final"),
            "game_adapted_checkpoint": str(game_root / "final"),
            "evaluation_summary": str(evaluation_summary),
            "report_artifact_manifest": str(report_manifest),
        },
        "claim_boundary": (
            "The custom-model study compares controlled training stages. "
            "It does not replace the grounded Qwen3-4B deployment result, and "
            "lightweight project-local pretraining is not full foundation-model pretraining."
        ),
    }
    write_json(output_root / "study_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
