from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.training.train_sft import (
    normalize_target_modules,
    resolve_compute_dtype,
    resolve_model_reference,
    validate_training_config,
)


def test_training_config_validation_and_warmup_modes() -> None:
    validate_training_config(
        {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "warmup_ratio": 0.03,
        }
    )
    validate_training_config(
        {
            "per_device_train_batch_size": 1,
            "learning_rate": 1e-4,
            "warmup_steps": 0,
        }
    )
    with pytest.raises(ValueError, match="warmup_ratio"):
        validate_training_config({"warmup_ratio": 1.0})
    with pytest.raises(ValueError, match="learning_rate"):
        validate_training_config({"learning_rate": 0})


def test_dtype_target_modules_and_model_reference(tmp_path: Path) -> None:
    assert resolve_compute_dtype("fp32") is torch.float32
    assert normalize_target_modules("all-linear") == "all-linear"
    assert normalize_target_modules(["q_proj", "v_proj"]) == ["q_proj", "v_proj"]

    local_model = tmp_path / "model"
    local_model.mkdir()
    assert resolve_model_reference(local_model) == str(local_model.resolve())
    assert resolve_model_reference("Qwen/Qwen2.5-0.5B") == "Qwen/Qwen2.5-0.5B"
