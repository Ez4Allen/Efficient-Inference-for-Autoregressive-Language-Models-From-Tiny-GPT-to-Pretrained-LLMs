"""Supervised fine-tuning entry points.

Optional training dependencies are imported lazily so the core package remains usable
without PEFT or bitsandbytes.
"""


def train_sft_main() -> None:
    from .train_sft import main

    main()


__all__ = ["train_sft_main"]
