"""Configurable training loop for the tiny GPT-style language model."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from itertools import cycle
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from src.data.tiny_lm_dataset import TextDataset, create_train_val_split, load_text_file
from src.models.tiny_lm.model import TinyGPT
from src.models.tiny_lm.tokenizer import CharTokenizer
from src.utils.io import read_yaml, write_json
from src.utils.paths import DATA_ROOT, RESULTS_ROOT, resolve_project_path
from src.utils.seed import set_global_seed


@dataclass(frozen=True)
class TinyLMTrainingConfig:
    data_path: Path
    output_dir: Path
    context_length: int = 128
    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 128
    d_ff: int = 512
    dropout: float = 0.1
    batch_size: int = 32
    learning_rate: float = 5e-4
    max_steps: int = 1_000
    eval_interval: int = 100
    eval_batches: int = 20
    val_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "auto"
    num_workers: int = 0

    def validate(self) -> None:
        if self.context_length < 2:
            raise ValueError("context_length must be at least 2.")
        if (
            self.n_layers < 1
            or self.n_heads < 1
            or self.d_model < 1
            or self.d_ff < 1
        ):
            raise ValueError("Model dimensions must be positive.")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if self.batch_size < 1 or self.max_steps < 1:
            raise ValueError("batch_size and max_steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.eval_interval < 1 or self.eval_batches < 1:
            raise ValueError("eval_interval and eval_batches must be positive.")
        if not 0 < self.val_ratio < 1:
            raise ValueError("val_ratio must be between 0 and 1.")
        if self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive.")


def _dataset_path(dataset: str | Path) -> Path:
    if str(dataset) == "tiny_shakespeare":
        return DATA_ROOT / "tiny_shakespeare" / "input.txt"
    return resolve_project_path(dataset)


def load_training_config(path: str | Path) -> TinyLMTrainingConfig:
    """Load the existing ``tiny_gpt.yaml`` layout into a validated config."""

    payload = read_yaml(resolve_project_path(path))
    if not isinstance(payload, Mapping):
        raise TypeError("Tiny LM configuration must be a YAML mapping.")
    model = payload.get("model", {})
    training = payload.get("training", {})
    if not isinstance(model, Mapping) or not isinstance(training, Mapping):
        raise TypeError("model and training sections must be mappings.")

    experiment_name = str(payload.get("experiment_name", "tiny_gpt_shakespeare"))
    output_dir = training.get("output_dir", RESULTS_ROOT / experiment_name)
    config = TinyLMTrainingConfig(
        data_path=_dataset_path(training.get("dataset", "tiny_shakespeare")),
        output_dir=resolve_project_path(output_dir),
        context_length=int(model.get("context_length", 128)),
        n_layers=int(model.get("n_layers", 2)),
        n_heads=int(model.get("n_heads", 4)),
        d_model=int(model.get("d_model", 128)),
        d_ff=int(model.get("d_ff", 4 * int(model.get("d_model", 128)))),
        dropout=float(model.get("dropout", 0.1)),
        batch_size=int(training.get("batch_size", 32)),
        learning_rate=float(training.get("learning_rate", 5e-4)),
        max_steps=int(training.get("max_steps", 1_000)),
        eval_interval=int(training.get("eval_interval", 100)),
        eval_batches=int(training.get("eval_batches", 20)),
        val_ratio=float(training.get("val_ratio", 0.1)),
        weight_decay=float(training.get("weight_decay", 0.01)),
        max_grad_norm=float(training.get("max_grad_norm", 1.0)),
        seed=int(training.get("seed", 42)),
        device=str(training.get("device", "auto")),
        num_workers=int(training.get("num_workers", 0)),
    )
    config.validate()
    return config


def _resolve_device(value: str) -> torch.device:
    normalized = value.strip().casefold()
    if normalized in {"", "auto"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable.")
    return device


@torch.inference_mode()
def evaluate(
    model: TinyGPT,
    data_loader: DataLoader,
    device: torch.device,
    *,
    max_batches: int = 20,
) -> float:
    """Return the mean validation loss without changing the caller's train state."""

    was_training = model.training
    model.eval()
    losses: list[float] = []
    for batch_index, (inputs, targets) in enumerate(data_loader):
        if batch_index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _, loss = model(inputs, targets)
        if loss is not None:
            losses.append(float(loss.item()))
    model.train(was_training)
    if not losses:
        raise ValueError("Validation loader produced no batches.")
    return sum(losses) / len(losses)


def train_tiny_lm(config: TinyLMTrainingConfig) -> dict[str, Any]:
    """Train TinyGPT and save a checkpoint, tokenizer, and JSON report."""

    config.validate()
    if not config.data_path.exists():
        raise FileNotFoundError(f"Training text not found: {config.data_path}")
    set_global_seed(config.seed)
    device = _resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    text = load_text_file(config.data_path)
    tokenizer = CharTokenizer()
    tokenizer.fit(text)
    token_ids = tokenizer.encode(text)
    train_ids, validation_ids = create_train_val_split(
        token_ids, val_ratio=config.val_ratio
    )
    train_dataset = TextDataset(train_ids, config.context_length)
    validation_dataset = TextDataset(validation_ids, config.context_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    if not train_loader:
        raise ValueError("Training loader contains no full batches; lower batch_size.")

    model = TinyGPT(
        vocab_size=int(tokenizer.vocab_size),
        block_size=config.context_length,
        n_layer=config.n_layers,
        n_head=config.n_heads,
        n_embd=config.d_model,
        d_ff=config.d_ff,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    history: list[dict[str, Any]] = []
    start = perf_counter()
    model.train()
    for step, (inputs, targets) in enumerate(cycle(train_loader), start=1):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        _, loss = model(inputs, targets)
        if loss is None:
            raise RuntimeError("Model did not return a training loss.")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.max_grad_norm
        )
        optimizer.step()

        if step == 1 or step % config.eval_interval == 0 or step == config.max_steps:
            validation_loss = evaluate(
                model,
                validation_loader,
                device,
                max_batches=config.eval_batches,
            )
            entry = {
                "step": step,
                "train_loss": float(loss.item()),
                "validation_loss": validation_loss,
                "validation_perplexity": float(math.exp(min(validation_loss, 20.0))),
                "gradient_norm": float(grad_norm),
            }
            history.append(entry)
            print(json.dumps(entry))

        if step >= config.max_steps:
            break

    elapsed = perf_counter() - start
    checkpoint_path = config.output_dir / "model.pt"
    tokenizer_path = config.output_dir / "tokenizer.json"
    report_path = config.output_dir / "training_report.json"
    temporary_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "block_size": config.context_length,
                "n_layer": config.n_layers,
                "n_head": config.n_heads,
                "n_embd": config.d_model,
                "d_ff": config.d_ff,
                "dropout": config.dropout,
            },
            "training_config": {
                **asdict(config),
                "data_path": str(config.data_path),
                "output_dir": str(config.output_dir),
            },
            "history": history,
        },
        temporary_checkpoint,
    )
    temporary_checkpoint.replace(checkpoint_path)
    tokenizer.save(tokenizer_path)

    report = {
        "status": "passed",
        "device": str(device),
        "steps": config.max_steps,
        "elapsed_seconds": elapsed,
        "tokens": {
            "total": len(token_ids),
            "train": len(train_ids),
            "validation": len(validation_ids),
        },
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_path": str(checkpoint_path),
        "tokenizer_path": str(tokenizer_path),
        "history": history,
    }
    write_json(report_path, report)
    return report
