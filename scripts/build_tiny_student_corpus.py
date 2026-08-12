#!/usr/bin/env python3
"""Build the local multi-game causal-pretraining corpus for TinyQwenStudent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.project_pretraining_corpus import ProjectCorpusBuilder


def existing(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data/gameguide/custom_model/pretraining_corpus.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/gameguide/custom_model/pretraining_manifest.json",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-eval-entities",
        action="store_true",
        help="Disable entity-held-out filtering. Exact eval questions remain excluded.",
    )
    args = parser.parse_args()

    builder = ProjectCorpusBuilder(
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        exclude_eval_entities=not args.include_eval_entities,
    )
    builder.add_evaluation_records(
        existing(
            [
                PROJECT_ROOT / "data/stardew/evaluation/stardew_validation_v1.jsonl",
                PROJECT_ROOT / "data/stardew/evaluation/stardew_eval_v1.jsonl",
                PROJECT_ROOT / "data/terraria/terraria_validation_v1.jsonl",
                PROJECT_ROOT / "data/terraria/terraria_eval.jsonl",
            ]
        )
    )

    builder.add_chat_jsonl(
        PROJECT_ROOT / "data/stardew/sft/train.jsonl",
        domain="stardew_valley",
        license_name="project training data; source URLs retained per record",
    )
    builder.add_chat_jsonl(
        PROJECT_ROOT / "data/terraria/terraria_train_v1.jsonl",
        domain="terraria",
        license_name="project training data; source URLs retained per record",
    )

    stardew_guides = PROJECT_ROOT / "data/stardew/guides/seed/pages.jsonl"
    if stardew_guides.exists():
        builder.add_guide_pages(stardew_guides, domain="stardew_valley")

    terraria_guide_candidates = [
        PROJECT_ROOT / "data/terraria/guides/seed/pages.jsonl",
        PROJECT_ROOT / "data/terraria/guides/raw/pages.jsonl",
    ]
    for path in terraria_guide_candidates:
        if path.exists():
            builder.add_guide_pages(path, domain="terraria")
            break

    for path in existing(
        [
            PROJECT_ROOT / "data/stardew/catalog/cleaned/facts.jsonl",
            PROJECT_ROOT / "data/stardew/catalog/cleaned/crops.jsonl",
            PROJECT_ROOT / "data/stardew/catalog/cleaned/fish.jsonl",
            PROJECT_ROOT / "data/stardew/catalog/cleaned/villagers.jsonl",
            PROJECT_ROOT / "data/stardew/catalog/cleaned/recipes.jsonl",
            PROJECT_ROOT / "data/stardew/catalog/cleaned/bundles.jsonl",
            PROJECT_ROOT / "data/stardew/catalog/cleaned/acquisition_sources.jsonl",
        ]
    ):
        builder.add_catalog_jsonl(path, domain="stardew_valley")

    for path in existing(
        [
            PROJECT_ROOT / "data/terraria/catalog/cleaned/Items.jsonl",
            PROJECT_ROOT / "data/terraria/catalog/cleaned/NPCs.jsonl",
            PROJECT_ROOT / "data/terraria/catalog/cleaned/Recipes.jsonl",
            PROJECT_ROOT / "data/terraria/catalog/cleaned/Drops.jsonl",
        ]
    ):
        builder.add_catalog_jsonl(path, domain="terraria")

    manifest = builder.write(args.output, args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
