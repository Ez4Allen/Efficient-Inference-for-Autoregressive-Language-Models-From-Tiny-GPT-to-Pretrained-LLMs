# Stardew Valley regression suite v1

Files:

```text
stardew_validation_v1.jsonl  40 records
stardew_eval_v1.jsonl        60 records
manifest_v1.json             counts, distributions, hashes
```

The 100 cases are balanced 50/50 across English and Chinese and cover crops, fish, villagers, recipes, bundles, acquisition, and guide retrieval. Status targets are 70 `found`, 10 `needs_context`, 10 `partial`, and 10 `not_found`.

Run:

```bash
python scripts/evaluate_gameguidelm.py \
  --input data/stardew/evaluation/stardew_validation_v1.jsonl \
          data/stardew/evaluation/stardew_eval_v1.jsonl \
  --output results/stardew/evaluation_details.jsonl \
  --summary results/stardew/evaluation_summary.json
```

These are deterministic engineering regression candidates. They remain `machine_validated`, `reviewer=null`, and `human_review_required=true` until an independent person checks each source and answer.
