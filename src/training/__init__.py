"""Training entry points with optional dependencies imported lazily."""

from .tiny_qwen_draft import (
    TinyQwenDraftTrainingConfig,
    load_tiny_qwen_draft_training_config,
    train_tiny_qwen_draft,
)
from .tiny_qwen_pretraining import (
    TinyQwenPretrainingConfig,
    load_tiny_qwen_pretraining_config,
    pretrain_tiny_qwen_student,
)


def train_sft_main() -> None:
    from .train_sft import main

    main()


__all__ = [
    "TinyQwenDraftTrainingConfig",
    "TinyQwenPretrainingConfig",
    "load_tiny_qwen_draft_training_config",
    "load_tiny_qwen_pretraining_config",
    "pretrain_tiny_qwen_student",
    "train_sft_main",
    "train_tiny_qwen_draft",
]
