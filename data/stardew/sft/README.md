# Legacy Stardew AI-assisted SFT candidates

This directory preserves and audits the 1,262 records contributed in the original Stardew fork. They are useful candidate material, but the original `verified=true` field did not provide auditable proof of line-by-line human review.

The cleanup process therefore sets every retained record to:

```json
{
  "verified": false,
  "review_status": "pending",
  "reviewed_by": null
}
```

It also:

- canonicalizes Wiki URLs;
- maps 490 free-form topics to eight controlled intents while preserving the original topic;
- generates split-aligned stable IDs;
- keeps connected source-page and normalized-template groups in one split;
- reports zero cross-split source and template overlap;
- writes rejected records with explicit reasons instead of silently dropping them.

Run:

```bash
python scripts/clean_stardew_sft_data.py
```

The `train.jsonl`, `validation.jsonl`, and `eval.jsonl` files here are candidate-development partitions only. They are not the formal benchmark under `data/stardew/evaluation`, and they should not be used for training until reviewed.
