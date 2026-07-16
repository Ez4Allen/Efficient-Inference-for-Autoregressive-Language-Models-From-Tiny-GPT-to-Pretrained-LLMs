
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import torch
import yaml
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.data.sft_dataset import (
    SFTDataCollator,
    SFTJsonlDataset,
)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise TypeError(
            f"Configuration must be a YAML mapping: {path}"
        )

    return config


def require_section(
    config: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """Return a required configuration section."""

    section = config.get(section_name)

    if not isinstance(section, dict):
        raise KeyError(
            f"Missing or invalid config section: "
            f"{section_name!r}"
        )

    return section


def require_value(
    section: dict[str, Any],
    key: str,
    section_name: str,
) -> Any:
    """Return a required configuration value."""

    if key not in section:
        raise KeyError(
            f"Missing config value: "
            f"{section_name}.{key}"
        )

    return section[key]


def resolve_compute_dtype(
    value: str | None,
) -> torch.dtype:
    """
    Resolve the compute dtype used by 4-bit layers.

    auto:
        BF16 when supported, otherwise FP16.
    """

    normalized = str(value or "auto").lower()

    if normalized == "auto":
        if (
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ):
            return torch.bfloat16

        return torch.float16

    dtype_map = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }

    if normalized not in dtype_map:
        raise ValueError(
            f"Unsupported compute dtype: {value!r}"
        )

    return dtype_map[normalized]


def normalize_target_modules(
    value: Any,
) -> str | list[str]:
    """
    Validate LoRA target modules.

    Supported forms:
        "all-linear"
        ["q_proj", "k_proj", "v_proj", ...]
    """

    if isinstance(value, str):
        if not value.strip():
            raise ValueError(
                "lora.target_modules cannot be empty"
            )

        return value.strip()

    if isinstance(value, list):
        modules = [
            str(module).strip()
            for module in value
            if str(module).strip()
        ]

        if not modules:
            raise ValueError(
                "lora.target_modules list cannot be empty"
            )

        return modules

    raise TypeError(
        "lora.target_modules must be a string or list"
    )


def count_parameters(
    model: torch.nn.Module,
) -> tuple[int, int]:
    """Return trainable and total parameter counts."""

    trainable = 0
    total = 0

    for parameter in model.parameters():
        parameter_count = parameter.numel()
        total += parameter_count

        if parameter.requires_grad:
            trainable += parameter_count

    return trainable, total


def print_parameter_summary(
    model: torch.nn.Module,
) -> None:
    """Print LoRA trainable-parameter statistics."""

    trainable, total = count_parameters(model)

    percentage = (
        100.0 * trainable / total
        if total > 0
        else 0.0
    )

    print("\n=== Parameter Summary ===")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Total parameters:     {total:,}")
    print(f"Trainable percentage: {percentage:.4f}%")


def build_training_arguments(
    training_config: dict[str, Any],
    compute_dtype: torch.dtype,
    has_validation: bool,
) -> TrainingArguments:
    """
    Build TrainingArguments.

    The eval argument name changed across Transformers versions.
    This function supports both eval_strategy and
    evaluation_strategy.
    """

    output_dir = str(
        require_value(
            training_config,
            "output_dir",
            "training",
        )
    )

    max_steps = int(
        training_config.get("max_steps", -1)
    )

    eval_strategy = (
        "steps"
        if has_validation
        else "no"
    )

    arguments: dict[str, Any] = {
        "output_dir": output_dir,
        "num_train_epochs": float(
            training_config.get(
                "num_train_epochs",
                1.0,
            )
        ),
        "max_steps": max_steps,
        "per_device_train_batch_size": int(
            training_config.get(
                "per_device_train_batch_size",
                1,
            )
        ),
        "per_device_eval_batch_size": int(
            training_config.get(
                "per_device_eval_batch_size",
                1,
            )
        ),
        "gradient_accumulation_steps": int(
            training_config.get(
                "gradient_accumulation_steps",
                1,
            )
        ),
        "learning_rate": float(
            training_config.get(
                "learning_rate",
                1e-4,
            )
        ),
        "weight_decay": float(
            training_config.get(
                "weight_decay",
                0.0,
            )
        ),
        "warmup_steps": float(
            training_config.get(
                "warmup_steps",
                training_config.get(
                    "warmup_ratio",
                    0.0,
                ),
            )
        ),
        "lr_scheduler_type": str(
            training_config.get(
                "lr_scheduler_type",
                "linear",
            )
        ),
        "max_grad_norm": float(
            training_config.get(
                "max_grad_norm",
                1.0,
            )
        ),
        "logging_strategy": "steps",
        "logging_steps": int(
            training_config.get(
                "logging_steps",
                1,
            )
        ),
        "save_strategy": "steps",
        "save_steps": int(
            training_config.get(
                "save_steps",
                100,
            )
        ),
        "save_total_limit": int(
            training_config.get(
                "save_total_limit",
                2,
            )
        ),
        "eval_steps": int(
            training_config.get(
                "eval_steps",
                100,
            )
        ),
        "gradient_checkpointing": bool(
            training_config.get(
                "gradient_checkpointing",
                True,
            )
        ),
        "optim": str(
            training_config.get(
                "optim",
                "paged_adamw_8bit",
            )
        ),
        "bf16": compute_dtype == torch.bfloat16,
        "fp16": compute_dtype == torch.float16,
        "report_to": "none",
        "remove_unused_columns": False,
        "prediction_loss_only": True,
        "seed": int(
            training_config.get("seed", 42)
        ),
        "data_seed": int(
            training_config.get("seed", 42)
        ),
        "dataloader_num_workers": int(
            training_config.get(
                "dataloader_num_workers",
                0,
            )
        ),
        "save_safetensors": True,
    }

    signature = inspect.signature(
        TrainingArguments.__init__
    )

    if "eval_strategy" in signature.parameters:
        arguments["eval_strategy"] = eval_strategy

    elif "evaluation_strategy" in signature.parameters:
        arguments["evaluation_strategy"] = eval_strategy

    else:
        if has_validation:
            raise RuntimeError(
                "This Transformers version does not expose "
                "an evaluation strategy argument"
            )

    # Keep compatibility across Transformers versions.
    supported_parameters = set(
        signature.parameters
    )

    unsupported_arguments = sorted(
        set(arguments) - supported_parameters
    )

    if unsupported_arguments:
        print(
            "Ignoring unsupported TrainingArguments: "
            + ", ".join(unsupported_arguments)
        )

    arguments = {
        key: value
        for key, value in arguments.items()
        if key in supported_parameters
    }

    return TrainingArguments(**arguments)


def load_tokenizer(
    model_name_or_path: str,
    trust_remote_code: bool,
):
    """Load and configure the tokenizer."""

    local_files_only = Path(
        model_name_or_path
    ).exists()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(
                "Tokenizer has neither pad_token_id "
                "nor eos_token_id"
            )

        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def load_qlora_model(
    model_config: dict[str, Any],
    quantization_config: dict[str, Any],
    lora_config: dict[str, Any],
    training_config: dict[str, Any],
    compute_dtype: torch.dtype,
):
    """Load a 4-bit base model and attach LoRA adapters."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA training requires a CUDA GPU "
            "in this training script"
        )

    model_name_or_path = str(
        require_value(
            model_config,
            "model_name_or_path",
            "model",
        )
    )

    trust_remote_code = bool(
        model_config.get(
            "trust_remote_code",
            True,
        )
    )

    load_in_4bit = bool(
        quantization_config.get(
            "load_in_4bit",
            True,
        )
    )

    if not load_in_4bit:
        raise ValueError(
            "This script currently expects "
            "quantization.load_in_4bit=true"
        )

    bitsandbytes_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=str(
            quantization_config.get(
                "bnb_4bit_quant_type",
                "nf4",
            )
        ),
        bnb_4bit_use_double_quant=bool(
            quantization_config.get(
                "bnb_4bit_use_double_quant",
                True,
            )
        ),
        bnb_4bit_compute_dtype=compute_dtype,
    )

    local_files_only = Path(
        model_name_or_path
    ).exists()

    device_index = torch.cuda.current_device()

    print("\n=== Loading Base Model ===")
    print(f"Model: {model_name_or_path}")
    print(f"CUDA device: {device_index}")
    print(f"Compute dtype: {compute_dtype}")
    print("Quantization: 4-bit")

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
        quantization_config=bitsandbytes_config,
        torch_dtype=compute_dtype,
        device_map={"": device_index},
        low_cpu_mem_usage=True,
    )

    gradient_checkpointing = bool(
        training_config.get(
            "gradient_checkpointing",
            True,
        )
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=(
            gradient_checkpointing
        ),
    )

    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(
            lora_config.get("r", 16)
        ),
        lora_alpha=int(
            lora_config.get("alpha", 32)
        ),
        lora_dropout=float(
            lora_config.get("dropout", 0.05)
        ),
        target_modules=normalize_target_modules(
            lora_config.get(
                "target_modules",
                "all-linear",
            )
        ),
        bias=str(
            lora_config.get("bias", "none")
        ),
    )

    model = get_peft_model(
        model,
        peft_config,
    )

    print_parameter_summary(model)

    return model


def save_run_metadata(
    output_dir: Path,
    config: dict[str, Any],
    config_path: Path,
    train_size: int,
    validation_size: int | None,
    model: torch.nn.Module,
) -> None:
    """Save resolved experiment metadata."""

    trainable, total = count_parameters(model)

    metadata = {
        "config_path": str(config_path),
        "train_size": train_size,
        "validation_size": validation_size,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percentage": (
            100.0 * trainable / total
            if total > 0
            else 0.0
        ),
        "config": config,
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        output_dir / "run_metadata.json"
    )

    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2,
        )


def train(
    config_path: str | Path,
    resume_from_checkpoint: str | None = None,
) -> None:
    """Run one generic QLoRA SFT experiment."""

    config_path = Path(config_path)
    config = load_yaml(config_path)

    model_config = require_section(
        config,
        "model",
    )
    data_config = require_section(
        config,
        "data",
    )
    quantization_config = require_section(
        config,
        "quantization",
    )
    lora_config = require_section(
        config,
        "lora",
    )
    training_config = require_section(
        config,
        "training",
    )

    seed = int(
        training_config.get("seed", 42)
    )
    set_seed(seed)

    model_name_or_path = str(
        require_value(
            model_config,
            "model_name_or_path",
            "model",
        )
    )

    trust_remote_code = bool(
        model_config.get(
            "trust_remote_code",
            True,
        )
    )

    compute_dtype = resolve_compute_dtype(
        quantization_config.get(
            "compute_dtype",
            "auto",
        )
    )

    tokenizer = load_tokenizer(
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
    )

    train_path = str(
        require_value(
            data_config,
            "train_path",
            "data",
        )
    )

    validation_path_value = data_config.get(
        "validation_path"
    )

    max_length = int(
        data_config.get(
            "max_length",
            512,
        )
    )

    print("\n=== Loading Datasets ===")

    train_dataset = SFTJsonlDataset(
        path=train_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    validation_dataset = None

    if validation_path_value:
        validation_dataset = SFTJsonlDataset(
            path=str(validation_path_value),
            tokenizer=tokenizer,
            max_length=max_length,
        )

    print(
        f"Train examples: "
        f"{len(train_dataset)}"
    )

    if validation_dataset is not None:
        print(
            f"Validation examples: "
            f"{len(validation_dataset)}"
        )

    model = load_qlora_model(
        model_config=model_config,
        quantization_config=quantization_config,
        lora_config=lora_config,
        training_config=training_config,
        compute_dtype=compute_dtype,
    )

    data_collator = SFTDataCollator(
        tokenizer=tokenizer,
    )

    training_arguments = build_training_arguments(
        training_config=training_config,
        compute_dtype=compute_dtype,
        has_validation=(
            validation_dataset is not None
        ),
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    output_dir = Path(
        training_arguments.output_dir
    )

    save_run_metadata(
        output_dir=output_dir,
        config=config,
        config_path=config_path,
        train_size=len(train_dataset),
        validation_size=(
            len(validation_dataset)
            if validation_dataset is not None
            else None
        ),
        model=model,
    )

    print("\n=== Starting Training ===")

    train_result = trainer.train(
        resume_from_checkpoint=(
            resume_from_checkpoint
        )
    )

    print("\n=== Final Evaluation ===")

    evaluation_metrics = None

    if validation_dataset is not None:
        evaluation_metrics = trainer.evaluate()
        print(evaluation_metrics)

    final_adapter_dir = (
        output_dir / "final_adapter"
    )

    final_adapter_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(final_adapter_dir)
    )

    tokenizer.save_pretrained(
        str(final_adapter_dir)
    )

    trainer.save_state()

    metrics = {
        "train": train_result.metrics,
        "evaluation": evaluation_metrics,
    }

    with (
        output_dir / "final_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== Training Complete ===")
    print(
        f"Adapter saved to: "
        f"{final_adapter_dir}"
    )
    print(
        f"Metrics saved to: "
        f"{output_dir / 'final_metrics.json'}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic QLoRA supervised "
            "fine-tuning entry point"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML experiment config",
    )

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help=(
            "Optional Trainer checkpoint directory "
            "to resume from"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train(
        config_path=args.config,
        resume_from_checkpoint=(
            args.resume_from_checkpoint
        ),
    )


if __name__ == "__main__":
    main()
