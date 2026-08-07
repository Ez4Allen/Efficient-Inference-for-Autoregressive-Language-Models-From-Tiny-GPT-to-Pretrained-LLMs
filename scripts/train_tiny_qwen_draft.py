#!/usr/bin/env python3
"""Train the custom Qwen-token-compatible draft model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.tiny_qwen_draft import (
    load_tiny_qwen_draft_training_config,
    train_tiny_qwen_draft,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tiny_qwen_draft.yaml",
    )
    args = parser.parse_args()
    config = load_tiny_qwen_draft_training_config(args.config)
    report = train_tiny_qwen_draft(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
