# Speculative-decoding correctness modes

GameGuideLM exposes two greedy speculative-verification modes because a target
model's multi-token verification path is not always bitwise equivalent to its
one-token decode path in BF16/FP16 GPU execution.

## `exact`

```yaml
generation:
  verification_mode: exact
```

The target checks candidate tokens through the same `q_len=1` incremental path
used by target-only greedy decoding. For a deterministic target in evaluation
mode, this path guarantees token identity with the target-only reference and
measures draft acceptance without conflating it with query-length-dependent
target kernels.

`exact` is a correctness mode. It intentionally performs approximately one
target decode step per committed token, so it should not be used to claim a
speculative speedup.

## `block`

```yaml
generation:
  verification_mode: block
```

The target validates the full candidate block in one forward call. This is the
performance-oriented speculative path: a strong draft can reduce target forward
calls. The benchmark must report `exact_target_match_rate` because the target's
`q_len>1` attention/cache computation can differ numerically from its
`q_len=1` greedy path.

A valid performance result reports both:

- wall-clock latency / tokens per second;
- token-level exact match against the target-only greedy reference.

Do not describe a block-mode run as target-token preserving when its exact-match
rate is below 1.0. The benchmark also reports the first mismatching token,
position-wise token agreement, and whether repeated target-only runs are
deterministic.

The model endpoint accepts an optional Hugging Face attention backend:

```yaml
models:
  target:
    attn_implementation: eager
```

Use this only as a controlled diagnostic rerun when block mode diverges. Keep
the dtype, prompts, generation length, and warm-up schedule fixed, and do not
compare latency across attention backends as though they were the same system.

## Target block-consistency diagnostic

Run the target by itself:

```bash
python scripts/diagnose_speculative_consistency.py \
  --config configs/gameguidelm_qwen3_pair.yaml \
  --prompt "Explain when and where I can catch Catfish." \
  --max-new-tokens 64 \
  --block-sizes 1 2 4 6 8 \
  --output results/model_study/block_consistency.json
```

Interpretation:

- block size 1 has zero mismatches, while larger blocks have mismatches: the
  target checkpoint/kernel is query-length sensitive in the selected dtype and
  attention backend;
- every block size has zero mismatches, but block speculative output diverges:
  investigate cache rollback/synchronization;
- `exact` mode diverges from target-only: treat this as a correctness failure.

## Benchmark commands

Correctness and acceptance:

```bash
python scripts/benchmark_gameguidelm.py \
  --config configs/gameguidelm_qwen3_pair.yaml \
  --input data/stardew/evaluation/stardew_eval_v1.jsonl \
  --output results/model_study/exact_rows.jsonl \
  --summary results/model_study/exact_summary.json \
  --engines target speculative \
  --verification-mode exact \
  --runs 3 \
  --limit 8
```

Performance-oriented block verification:

```bash
python scripts/benchmark_gameguidelm.py \
  --config configs/gameguidelm_qwen3_pair.yaml \
  --input data/stardew/evaluation/stardew_eval_v1.jsonl \
  --output results/model_study/block_rows.jsonl \
  --summary results/model_study/block_summary.json \
  --engines target speculative \
  --verification-mode block \
  --runs 5 \
  --limit 16
```

The benchmark compares generated token IDs rather than decoded strings.
