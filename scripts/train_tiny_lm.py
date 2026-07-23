"""Train the tiny GPT model from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.tiny_lm.train import load_training_config, train_tiny_lm  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "tiny_gpt.yaml",
    )
    args = parser.parse_args()
    report = train_tiny_lm(load_training_config(args.config))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
