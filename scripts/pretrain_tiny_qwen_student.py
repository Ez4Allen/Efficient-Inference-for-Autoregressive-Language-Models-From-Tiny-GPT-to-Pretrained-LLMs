#!/usr/bin/env python3
"""Run lightweight causal pretraining for TinyQwenStudent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.training.tiny_qwen_pretraining import (
    load_tiny_qwen_pretraining_config,
    pretrain_tiny_qwen_student,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_tiny_qwen_pretraining_config(args.config)
    report = pretrain_tiny_qwen_student(config)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
