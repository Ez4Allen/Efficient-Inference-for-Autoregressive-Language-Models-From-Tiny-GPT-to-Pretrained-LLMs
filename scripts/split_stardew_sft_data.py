#!/usr/bin/env python3
"""Compatibility wrapper for the audited Stardew SFT candidate splitter.

The legacy splitter grouped only by source page and mishandled small datasets.
This wrapper delegates to ``clean_stardew_sft_data.py`` so schema normalization,
review-state correction, connected source/template grouping, stable IDs, and
fraction validation happen together.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clean_stardew_sft_data import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--eval-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = run(
        args.input.expanduser().resolve(),
        args.outdir.expanduser().resolve(),
        val_fraction=args.val_frac,
        eval_fraction=args.eval_frac,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
