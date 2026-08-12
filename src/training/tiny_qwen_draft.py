"""From-scratch sequence distillation for the custom TinyQwenDraft model."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from src.data.sft_dataset import SFTDataCollator, SFTJsonlDataset, load_jsonl
from src.models.tiny_qwen_draft import TinyQwenDraft, TinyQwenDraftConfig
from src.models.tokenizer_contract import (
    chat_template_sha256,
    tokenizer_sha256,
    tokenizer_vocabulary_size,
)
from src.utils.device import resolve_device
from src.utils.io import read_yaml, write_json
from src.utils.paths import resolve_project_path
from src.utils.seed import set_global_seed


@dataclass(frozen=True)
class TinyQwenDraftTrainingConfig:
    tokenizer_name_or_path: str
    target_model_name_or_path: str
    train_path: Path
    validation_path: Path | None
    output_dir: Path
    initial_checkpoint: Path | None = None
    stage_name: str = "sequence_distillation"

    max_length: int = 512
    truncation_mode: str = "preserve_assistant"
    hidden_size: int = 256
    intermediate_size: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 4096
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1.0e-6
    attention_dropout: float = 0.0
    initializer_range: float = 0.02

    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    max_steps: int = 1_000
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    logging_steps: int = 10
    eval_steps: int = 100
    eval_batches: int = 20
    save_steps: int = 250
    seed: int = 42
    device: str = "auto"
    compute_dtype: str = "auto"
    num_workers: int = 0
    local_files_only: bool = False

    def validate(self) -> None:
        if not self.tokenizer_name_or_path.strip():
            raise ValueError("tokenizer_name_or_path is required.")
        if not self.target_model_name_or_path.strip():
            raise ValueError("target_model_name_or_path is required.")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2.")
        if self.truncation_mode not in {"right", "preserve_assistant"}:
            raise ValueError(
                "truncation_mode must be 'right' or 'preserve_assistant'."
            )
        if not self.stage_name.strip():
            raise ValueError("stage_name cannot be empty.")
        if self.max_length > self.max_position_embeddings:
            raise ValueError(
                "max_length cannot exceed max_position_embeddings."
            )
        if self.per_device_train_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be positive.")
        if self.per_device_eval_batch_size < 1:
            raise ValueError("per_device_eval_batch_size must be positive.")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1).")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")
        for name in ("logging_steps", "eval_steps", "eval_batches", "save_steps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")

        TinyQwenDraftConfig(
            vocab_size=2,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            max_position_embeddings=self.max_position_embeddings,
            rope_theta=self.rope_theta,
            rms_norm_eps=self.rms_norm_eps,
            attention_dropout=self.attention_dropout,
            initializer_range=self.initializer_range,
        ).validate()


def _required_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"Missing or invalid config section: {name}")
    return value


def _required_text(payload: Mapping[str, Any], key: str, section: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"{section}.{key} is required.")
    return value


def load_tiny_qwen_draft_training_config(
    path: str | Path,
) -> TinyQwenDraftTrainingConfig:
    payload = read_yaml(resolve_project_path(path))
    if not isinstance(payload, Mapping):
        raise TypeError("TinyQwenDraft training config must be a YAML mapping.")

    model = _required_mapping(payload, "model")
    data = _required_mapping(payload, "data")
    training = _required_mapping(payload, "training")

    validation_value = data.get("validation_path")
    validation_path = (
        resolve_project_path(validation_value) if validation_value else None
    )
    initial_value = model.get("initial_checkpoint")

    config = TinyQwenDraftTrainingConfig(
        tokenizer_name_or_path=_required_text(
            model,
            "tokenizer_name_or_path",
            "model",
        ),
        target_model_name_or_path=str(
            model.get(
                "target_model_name_or_path",
                model.get("tokenizer_name_or_path", ""),
            )
        ).strip(),
        train_path=resolve_project_path(
            _required_text(data, "train_path", "data")
        ),
        validation_path=validation_path,
        output_dir=resolve_project_path(
            _required_text(training, "output_dir", "training")
        ),
        initial_checkpoint=(
            resolve_project_path(initial_value) if initial_value else None
        ),
        stage_name=str(training.get("stage_name", "sequence_distillation")).strip(),
        max_length=int(data.get("max_length", 512)),
        truncation_mode=str(
            data.get("truncation_mode", "preserve_assistant")
        ).strip().casefold(),
        hidden_size=int(model.get("hidden_size", 256)),
        intermediate_size=int(model.get("intermediate_size", 768)),
        num_hidden_layers=int(model.get("num_hidden_layers", 6)),
        num_attention_heads=int(model.get("num_attention_heads", 4)),
        num_key_value_heads=int(model.get("num_key_value_heads", 2)),
        max_position_embeddings=int(model.get("max_position_embeddings", 4096)),
        rope_theta=float(model.get("rope_theta", 1_000_000.0)),
        rms_norm_eps=float(model.get("rms_norm_eps", 1.0e-6)),
        attention_dropout=float(model.get("attention_dropout", 0.0)),
        initializer_range=float(model.get("initializer_range", 0.02)),
        per_device_train_batch_size=int(
            training.get("per_device_train_batch_size", 1)
        ),
        per_device_eval_batch_size=int(
            training.get("per_device_eval_batch_size", 1)
        ),
        gradient_accumulation_steps=int(
            training.get("gradient_accumulation_steps", 8)
        ),
        learning_rate=float(training.get("learning_rate", 3.0e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        max_steps=int(training.get("max_steps", 1_000)),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        logging_steps=int(training.get("logging_steps", 10)),
        eval_steps=int(training.get("eval_steps", 100)),
        eval_batches=int(training.get("eval_batches", 20)),
        save_steps=int(training.get("save_steps", 250)),
        seed=int(training.get("seed", 42)),
        device=str(training.get("device", "auto")),
        compute_dtype=str(training.get("compute_dtype", "auto")),
        num_workers=int(training.get("num_workers", 0)),
        local_files_only=bool(model.get("local_files_only", False)),
    )
    config.validate()
    return config


def _compute_dtype(name: str, device: torch.device) -> torch.dtype:
    normalized = str(name).strip().casefold()
    if normalized in {"", "auto"}:
        if device.type == "cuda":
            return (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        return torch.float32
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported compute_dtype: {name!r}")
    result = mapping[normalized]
    if device.type != "cuda" and result != torch.float32:
        raise ValueError("CPU training requires compute_dtype=fp32.")
    return result


def _lr_multiplier(step: int, *, max_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    remaining = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
    return max(0.0, 1.0 - progress)


def _autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


@torch.inference_mode()
def evaluate_tiny_qwen_draft(
    model: TinyQwenDraft,
    loader: DataLoader,
    *,
    device: torch.device,
    compute_dtype: torch.dtype,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
        with _autocast_context(device, compute_dtype):
            output = model(
                **batch,
                use_cache=False,
                loss_only=True,
            )
        if output.loss is None:
            raise RuntimeError("TinyQwenDraft did not return an evaluation loss.")
        losses.append(float(output.loss.item()))
    model.train()
    if not losses:
        raise ValueError("Validation loader produced no batches.")
    return sum(losses) / len(losses)


def _serialize_training_config(
    config: TinyQwenDraftTrainingConfig,
) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("train_path", "validation_path", "output_dir"):
        value = payload[key]
        payload[key] = str(value) if value is not None else None
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_distillation_split(
    path: Path,
    *,
    expected_split: str,
) -> list[dict[str, Any]]:
    """Reject formal evaluation data and mislabeled distillation records."""

    normalized_parts = {part.casefold() for part in path.parts}
    normalized_stem = path.stem.casefold()
    if "evaluation" in normalized_parts or normalized_stem in {"eval", "test"}:
        raise ValueError(
            f"Refusing to train from a formal evaluation path: {path}. "
            "Generate a separate target-teacher train/validation file instead."
        )

    records = load_jsonl(path)
    expected = expected_split.casefold()
    for index, record in enumerate(records, start=1):
        declared = str(record.get("split", "")).strip().casefold()
        if declared != expected:
            record_id = record.get("id", f"line {index}")
            raise ValueError(
                f"{path}: record {record_id!r} declares split {declared!r}; "
                f"expected {expected!r}. Formal eval/test records must never "
                "enter TinyQwenDraft adaptation."
            )
    return records


def _target_source_distribution(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for record in records:
        source = str(record.get("target_source") or "unspecified")
        distribution[source] = distribution.get(source, 0) + 1
    return dict(sorted(distribution.items()))


def _environment_report(device: torch.device) -> dict[str, Any]:
    gpu_name = None
    if device.type == "cuda":
        index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        gpu_name = torch.cuda.get_device_name(index)
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": gpu_name,
    }


def train_tiny_qwen_draft(
    config: TinyQwenDraftTrainingConfig,
) -> dict[str, Any]:
    """Train a custom draft on target-generated chat continuations."""

    config.validate()
    if not config.train_path.exists():
        raise FileNotFoundError(f"Training data not found: {config.train_path}")
    if config.validation_path is not None and not config.validation_path.exists():
        raise FileNotFoundError(
            f"Validation data not found: {config.validation_path}"
        )
    if config.initial_checkpoint is not None and not config.initial_checkpoint.exists():
        raise FileNotFoundError(
            f"Initial checkpoint not found: {config.initial_checkpoint}"
        )

    train_records = _validate_distillation_split(
        config.train_path,
        expected_split="train",
    )
    validation_records = (
        _validate_distillation_split(
            config.validation_path,
            expected_split="validation",
        )
        if config.validation_path is not None
        else []
    )

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "TinyQwenDraft training requires Transformers. Install dependencies "
            "with `pip install -r requirements-training.txt`."
        ) from error

    set_global_seed(config.seed)
    device = resolve_device(config.device)
    compute_dtype = _compute_dtype(config.compute_dtype, device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.tokenizer_name_or_path,
        use_fast=True,
        local_files_only=config.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither PAD nor EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    model_config = TinyQwenDraftConfig(
        vocab_size=tokenizer_vocabulary_size(tokenizer),
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        attention_dropout=config.attention_dropout,
        initializer_range=config.initializer_range,
        tie_word_embeddings=True,
        use_cache=True,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        unk_token_id=tokenizer.unk_token_id,
        tokenizer_name_or_path=config.tokenizer_name_or_path,
        target_model_name_or_path=config.target_model_name_or_path,
        teacher_model_name_or_path=config.target_model_name_or_path,
        training_stage=config.stage_name,
        parent_checkpoint=(
            str(config.initial_checkpoint) if config.initial_checkpoint else None
        ),
        tokenizer_sha256=tokenizer_sha256(tokenizer),
        chat_template_sha256=chat_template_sha256(tokenizer),
    )
    model_config.validate()

    train_dataset = SFTJsonlDataset(
        config.train_path,
        tokenizer,
        max_length=config.max_length,
        truncation_mode=config.truncation_mode,
    )
    validation_dataset = (
        SFTJsonlDataset(
            config.validation_path,
            tokenizer,
            max_length=config.max_length,
            truncation_mode=config.truncation_mode,
        )
        if config.validation_path is not None
        else None
    )
    collator = SFTDataCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=config.per_device_eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=collator,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
        )
        if validation_dataset is not None
        else None
    )

    if config.initial_checkpoint is None:
        model = TinyQwenDraft(model_config)
        initialization = "random_initialization"
    else:
        model = TinyQwenDraft.from_pretrained(
            config.initial_checkpoint,
            map_location="cpu",
        )
        architecture_fields = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "max_position_embeddings",
        )
        mismatches = {
            field: (getattr(model.config, field), getattr(model_config, field))
            for field in architecture_fields
            if getattr(model.config, field) != getattr(model_config, field)
        }
        if mismatches:
            raise ValueError(
                "Initial checkpoint architecture mismatch: "
                + json.dumps(mismatches, sort_keys=True)
            )
        if model.config.tokenizer_sha256 != model_config.tokenizer_sha256:
            raise ValueError("Initial checkpoint tokenizer fingerprint mismatch.")
        model.config.target_model_name_or_path = config.target_model_name_or_path
        model.config.teacher_model_name_or_path = config.target_model_name_or_path
        model.config.training_stage = config.stage_name
        model.config.parent_checkpoint = str(config.initial_checkpoint)
        model.config.tokenizer_name_or_path = config.tokenizer_name_or_path
        model.config.chat_template_sha256 = model_config.chat_template_sha256
        model.config.validate()
        initialization = str(config.initial_checkpoint)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    warmup_steps = int(config.max_steps * config.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _lr_multiplier(
            step,
            max_steps=config.max_steps,
            warmup_steps=warmup_steps,
        ),
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=(device.type == "cuda" and compute_dtype == torch.float16)
    )

    history: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    optimizer_step = 0
    running_loss = 0.0
    start = perf_counter()
    model.train()

    for batch in cycle(train_loader):
        micro_step += 1
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
        with _autocast_context(device, compute_dtype):
            output = model(
                **batch,
                use_cache=False,
                loss_only=True,
            )
            if output.loss is None:
                raise RuntimeError("TinyQwenDraft did not return a training loss.")
            scaled_loss = output.loss / config.gradient_accumulation_steps

        scaler.scale(scaled_loss).backward()
        running_loss += float(output.loss.item())

        if micro_step % config.gradient_accumulation_steps != 0:
            continue

        scaler.unscale_(optimizer)
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        optimizer_step += 1

        mean_train_loss = running_loss / config.gradient_accumulation_steps
        running_loss = 0.0
        should_log = optimizer_step == 1 or optimizer_step % config.logging_steps == 0
        should_eval = (
            validation_loader is not None
            and (
                optimizer_step == 1
                or optimizer_step % config.eval_steps == 0
                or optimizer_step == config.max_steps
            )
        )

        entry: dict[str, Any] = {
            "step": optimizer_step,
            "train_loss": mean_train_loss,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "gradient_norm": float(gradient_norm),
        }
        if should_eval and validation_loader is not None:
            validation_loss = evaluate_tiny_qwen_draft(
                model,
                validation_loader,
                device=device,
                compute_dtype=compute_dtype,
                max_batches=config.eval_batches,
            )
            entry["validation_loss"] = validation_loss
            entry["validation_perplexity"] = float(
                math.exp(min(validation_loss, 20.0))
            )
        if should_log or should_eval:
            history.append(entry)
            print(json.dumps(entry, ensure_ascii=False))

        if optimizer_step % config.save_steps == 0:
            model.save_pretrained(
                config.output_dir / "checkpoints" / f"step_{optimizer_step:06d}",
            )

        if optimizer_step >= config.max_steps:
            break

    elapsed = perf_counter() - start
    final_directory = model.save_pretrained(
        config.output_dir / "final",
        tokenizer=tokenizer,
    )
    report = {
        "status": "passed",
        "stage": config.stage_name,
        "initialization": initialization,
        "model_type": model_config.model_type,
        "target_model_name_or_path": config.target_model_name_or_path,
        "tokenizer_name_or_path": config.tokenizer_name_or_path,
        "tokenizer_sha256": model_config.tokenizer_sha256,
        "chat_template_sha256": model_config.chat_template_sha256,
        "parameters": model.num_parameters(),
        "train_records": len(train_dataset),
        "validation_records": (
            len(validation_dataset) if validation_dataset is not None else 0
        ),
        "train_sha256": _file_sha256(config.train_path),
        "validation_sha256": (
            _file_sha256(config.validation_path)
            if config.validation_path is not None
            else None
        ),
        "train_target_sources": _target_source_distribution(train_records),
        "validation_target_sources": _target_source_distribution(
            validation_records
        ),
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "compute_dtype": str(compute_dtype),
        "environment": _environment_report(device),
        "final_checkpoint": str(final_directory),
        "architecture": model_config.to_dict(),
        "training_config": _serialize_training_config(config),
        "history": history,
        "warning": (
            "This is sequence-level teacher adaptation. Report model-pair "
            "alignment, diversity/generalization slices, speculative acceptance, "
            "and latency; training loss alone is not evidence of a useful model."
        ),
    }
    write_json(config.output_dir / "training_report.json", report)
    return report
