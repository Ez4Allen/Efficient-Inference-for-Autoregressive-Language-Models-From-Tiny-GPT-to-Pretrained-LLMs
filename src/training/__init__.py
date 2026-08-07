"""Training entry points with optional dependencies imported lazily."""

from .tiny_qwen_draft import (
    TinyQwenDraftTrainingConfig,
    load_tiny_qwen_draft_training_config,
    train_tiny_qwen_draft,
)


def train_sft_main() -> None:
    from .train_sft import main

    main()


__all__ = [
    "TinyQwenDraftTrainingConfig",
    "load_tiny_qwen_draft_training_config",
    "train_sft_main",
    "train_tiny_qwen_draft",
]
