"""Lightweight causal pretraining for the custom TinyQwenStudent model."""

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

from src.data.causal_pretraining import CausalDataCollator, CausalPackedDataset
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
class TinyQwenPretrainingConfig:
    tokenizer_name_or_path: str
    teacher_model_name_or_path: str
    corpus_path: Path
    output_dir: Path
    initial_checkpoint: Path | None = None

    max_length: int = 1024
    stride: int = 1024
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
    gradient_accumulation_steps: int = 16
    learning_rate: float = 3.0e-4
    weight_decay: float = 0.01
    max_steps: int = 1_000
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    logging_steps: int = 25
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
        if not self.teacher_model_name_or_path.strip():
            raise ValueError("teacher_model_name_or_path is required.")
        if self.max_length < 2:
            raise ValueError("max_length must be at least 2.")
        if self.max_length > self.max_position_embeddings:
            raise ValueError("max_length cannot exceed max_position_embeddings.")
        if self.stride < 1 or self.stride > self.max_length:
            raise ValueError("stride must be in [1, max_length].")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1).")
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


def _mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"Missing or invalid config section: {name}")
    return value


def _text(payload: Mapping[str, Any], key: str, section: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"{section}.{key} is required.")
    return value


def load_tiny_qwen_pretraining_config(path: str | Path) -> TinyQwenPretrainingConfig:
    payload = read_yaml(resolve_project_path(path))
    if not isinstance(payload, Mapping):
        raise TypeError("TinyQwen pretraining config must be a YAML mapping.")
    model = _mapping(payload, "model")
    data = _mapping(payload, "data")
    training = _mapping(payload, "training")
    initial = model.get("initial_checkpoint")
    config = TinyQwenPretrainingConfig(
        tokenizer_name_or_path=_text(model, "tokenizer_name_or_path", "model"),
        teacher_model_name_or_path=str(
            model.get("teacher_model_name_or_path")
            or model.get("target_model_name_or_path")
            or model.get("tokenizer_name_or_path")
            or ""
        ).strip(),
        corpus_path=resolve_project_path(_text(data, "corpus_path", "data")),
        output_dir=resolve_project_path(_text(training, "output_dir", "training")),
        initial_checkpoint=(resolve_project_path(initial) if initial else None),
        max_length=int(data.get("max_length", 1024)),
        stride=int(data.get("stride", data.get("max_length", 1024))),
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
        per_device_train_batch_size=int(training.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(training.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(training.get("gradient_accumulation_steps", 16)),
        learning_rate=float(training.get("learning_rate", 3.0e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        max_steps=int(training.get("max_steps", 1_000)),
        warmup_ratio=float(training.get("warmup_ratio", 0.03)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        logging_steps=int(training.get("logging_steps", 25)),
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
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
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
    dtype = mapping[normalized]
    if device.type != "cuda" and dtype != torch.float32:
        raise ValueError("CPU pretraining requires FP32.")
    return dtype


def _autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def _lr_multiplier(step: int, *, max_steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    remaining = max(1, max_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
    return max(0.0, 1.0 - progress)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment(device: torch.device) -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": (
            torch.cuda.get_device_name(device.index or 0)
            if device.type == "cuda"
            else None
        ),
    }


def _architecture_signature(config: TinyQwenDraftConfig) -> tuple[int, ...]:
    return (
        config.vocab_size,
        config.hidden_size,
        config.intermediate_size,
        config.num_hidden_layers,
        config.num_attention_heads,
        config.num_key_value_heads,
        config.max_position_embeddings,
    )


def _create_or_load_model(
    config: TinyQwenPretrainingConfig,
    model_config: TinyQwenDraftConfig,
    *,
    tokenizer: Any,
    device: torch.device,
) -> tuple[TinyQwenDraft, str]:
    if config.initial_checkpoint is None:
        return TinyQwenDraft(model_config).to(device), "random_initialization"
    model = TinyQwenDraft.from_pretrained(config.initial_checkpoint, map_location="cpu")
    if _architecture_signature(model.config) != _architecture_signature(model_config):
        raise ValueError(
            "initial_checkpoint architecture does not match the pretraining config."
        )
    if model.config.tokenizer_sha256 != model_config.tokenizer_sha256:
        raise ValueError("initial_checkpoint tokenizer fingerprint mismatch.")
    model.config.teacher_model_name_or_path = config.teacher_model_name_or_path
    model.config.target_model_name_or_path = config.teacher_model_name_or_path
    model.config.training_stage = "lightweight_causal_pretraining"
    model.config.parent_checkpoint = str(config.initial_checkpoint)
    model.config.tokenizer_name_or_path = config.tokenizer_name_or_path
    model.config.chat_template_sha256 = model_config.chat_template_sha256
    model.config.validate()
    return model.to(device), str(config.initial_checkpoint)


@torch.inference_mode()
def evaluate_causal_model(
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
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with _autocast(device, compute_dtype):
            output = model(**batch, use_cache=False, loss_only=True)
        if output.loss is None:
            raise RuntimeError("Causal model returned no loss.")
        losses.append(float(output.loss.item()))
    model.train()
    if not losses:
        raise RuntimeError("Validation loader produced no batches.")
    return sum(losses) / len(losses)


def pretrain_tiny_qwen_student(config: TinyQwenPretrainingConfig) -> dict[str, Any]:
    config.validate()
    if not config.corpus_path.exists():
        raise FileNotFoundError(f"Causal corpus not found: {config.corpus_path}")
    if config.initial_checkpoint is not None and not config.initial_checkpoint.exists():
        raise FileNotFoundError(
            f"initial_checkpoint not found: {config.initial_checkpoint}"
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Transformers is required for TinyQwen pretraining.") from error

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
        target_model_name_or_path=config.teacher_model_name_or_path,
        teacher_model_name_or_path=config.teacher_model_name_or_path,
        training_stage="lightweight_causal_pretraining",
        parent_checkpoint=(str(config.initial_checkpoint) if config.initial_checkpoint else None),
        tokenizer_sha256=tokenizer_sha256(tokenizer),
        chat_template_sha256=chat_template_sha256(tokenizer),
    )
    model_config.validate()

    train_dataset = CausalPackedDataset(
        config.corpus_path,
        tokenizer,
        split="train",
        max_length=config.max_length,
        stride=config.stride,
    )
    validation_dataset = CausalPackedDataset(
        config.corpus_path,
        tokenizer,
        split="validation",
        max_length=config.max_length,
        stride=config.stride,
    )
    collator = CausalDataCollator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.per_device_train_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, initialization = _create_or_load_model(
        config,
        model_config,
        tokenizer=tokenizer,
        device=device,
    )
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
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with _autocast(device, compute_dtype):
            output = model(**batch, use_cache=False, loss_only=True)
            if output.loss is None:
                raise RuntimeError("TinyQwenStudent returned no pretraining loss.")
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

        entry: dict[str, Any] = {
            "step": optimizer_step,
            "train_loss": running_loss / config.gradient_accumulation_steps,
            "learning_rate": float(scheduler.get_last_lr()[0]),
            "gradient_norm": float(gradient_norm),
        }
        running_loss = 0.0
        should_eval = (
            optimizer_step == 1
            or optimizer_step % config.eval_steps == 0
            or optimizer_step == config.max_steps
        )
        should_log = optimizer_step == 1 or optimizer_step % config.logging_steps == 0
        if should_eval:
            validation_loss = evaluate_causal_model(
                model,
                validation_loader,
                device=device,
                compute_dtype=compute_dtype,
                max_batches=config.eval_batches,
            )
            entry["validation_loss"] = validation_loss
            entry["validation_perplexity"] = float(math.exp(min(validation_loss, 20.0)))
        if should_log or should_eval:
            history.append(entry)
            print(json.dumps(entry, ensure_ascii=False))
        if optimizer_step % config.save_steps == 0:
            model.save_pretrained(
                config.output_dir / "checkpoints" / f"step_{optimizer_step:06d}"
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
        "stage": "lightweight_causal_pretraining",
        "claim_boundary": (
            "Project-local lightweight causal pretraining; this is not full-scale "
            "foundation-model pretraining."
        ),
        "initialization": initialization,
        "teacher_model_name_or_path": config.teacher_model_name_or_path,
        "tokenizer_name_or_path": config.tokenizer_name_or_path,
        "tokenizer_sha256": model.config.tokenizer_sha256,
        "chat_template_sha256": model.config.chat_template_sha256,
        "parameters": model.num_parameters(),
        "corpus_sha256": _file_sha256(config.corpus_path),
        "train_documents": len(train_dataset.records),
        "validation_documents": len(validation_dataset.records),
        "train_chunks": len(train_dataset),
        "validation_chunks": len(validation_dataset),
        "train_tokens": train_dataset.total_tokens,
        "validation_tokens": validation_dataset.total_tokens,
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "compute_dtype": str(compute_dtype),
        "environment": _environment(device),
        "final_checkpoint": str(final_directory),
        "architecture": model.config.to_dict(),
        "training_config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in asdict(config).items()
        },
        "history": history,
    }
    write_json(config.output_dir / "pretraining_report.json", report)
    return report
