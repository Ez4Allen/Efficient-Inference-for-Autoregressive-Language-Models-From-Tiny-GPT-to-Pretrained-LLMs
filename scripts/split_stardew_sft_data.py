"""
Split human-reviewed SFT candidate records into train/validation/eval sets,
following the leakage rules in DATA_FORMAT.md section 6.

Key idea: split by SOURCE PAGE (source_urls), not by individual record.
This keeps every QA pair generated from the same wiki page inside the same
split, so a paraphrased duplicate can't leak between train and eval.

Usage:
    python split_sft_data.py \\
        --input sft_data/stardew_valley/reviewed.jsonl \\
        --outdir sft_data/stardew_valley \\
        --val-frac 0.1 --eval-frac 0.1

Input record requirements: same schema as generate_sft_candidates.py output,
but only records with "verified": true should be included (see reviewed.jsonl
note below).
"""

import json
import random
import argparse
import os
from collections import defaultdict


def load_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to reviewed, verified=true JSONL")
    ap.add_argument("--outdir", required=True, help="Directory to write train/validation/eval.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = load_jsonl(args.input)

    unverified = [r for r in records if not r.get("verified")]
    if unverified:
        print(f"WARNING: {len(unverified)} records are not verified=true. "
              f"They will be excluded from the split. Review them first.")
        records = [r for r in records if r.get("verified")]

    # Group by source page so all QA pairs from one page land in one split.
    # Source URLs are lowercased before grouping: the same wiki page has been
    # cited with inconsistent capitalization across records (e.g. Sunflower_Seeds
    # vs Sunflower_seeds), which would otherwise split one page's QA pairs
    # across train/validation/eval and defeat this leakage guard.
    groups = defaultdict(list)
    for r in records:
        key = tuple(sorted(u.lower() for u in r.get("source_urls", ["__no_source__"])))
        groups[key].append(r)

    group_keys = list(groups.keys())
    random.seed(args.seed)
    random.shuffle(group_keys)

    n_groups = len(group_keys)
    n_eval = max(1, int(n_groups * args.eval_frac))
    n_val = max(1, int(n_groups * args.val_frac))

    eval_keys = set(group_keys[:n_eval])
    val_keys = set(group_keys[n_eval:n_eval + n_val])
    train_keys = set(group_keys[n_eval + n_val:])

    train, val, evalset = [], [], []
    for key, recs in groups.items():
        for r in recs:
            if key in eval_keys:
                r["split"] = "eval"
                evalset.append(r)
            elif key in val_keys:
                r["split"] = "validation"
                val.append(r)
            else:
                r["split"] = "train"
                train.append(r)

    os.makedirs(args.outdir, exist_ok=True)
    write_jsonl(os.path.join(args.outdir, "train.jsonl"), train)
    write_jsonl(os.path.join(args.outdir, "validation.jsonl"), val)
    write_jsonl(os.path.join(args.outdir, "eval.jsonl"), evalset)

    print(f"Pages: {n_groups} total -> train {n_groups - n_eval - n_val}, "
          f"validation {n_val}, eval {n_eval}")
    print(f"Records: train {len(train)}, validation {len(val)}, eval {len(evalset)}")
    print(f"Written to {args.outdir}/")


if __name__ == "__main__":
    main()
