# Reproducibility

## Offline software validation

```bash
pip install -r requirements-dev.txt
python scripts/validate_release.py
```

This compiles the Python source, runs the complete test suite, rebuilds the
Terraria and Stardew structured SQLite stores from tracked snapshots, and runs
representative deterministic queries for both games. It does not access the
network or download model weights.

## Guide corpora

Wiki guide corpora are intentionally not committed. Build them locally:

```bash
python scripts/build_terraria_guides.py
python scripts/build_stardew_guides.py
```

After the initial online build, use `--offline` to rebuild indexes from local
raw snapshots.

## GPU model experiments

Install training/model extras:

```bash
pip install -r requirements-training.txt
```

The default paired checkpoints are:

```text
Draft:  Qwen/Qwen3-0.6B
Target: Qwen/Qwen3-4B
```

Run the pair smoke test before collecting performance results:

```bash
python scripts/smoke_qwen_pair.py \
  --config configs/gameguidelm_qwen3_pair.yaml \
  --prompt "Explain why draft and target tokenizers must match." \
  --max-new-tokens 32 \
  --engines draft target
```

Performance claims should be based on warm, repeated runs in one process. Model
download, checkpoint loading, and first-use CUDA initialization must not be
mixed into steady-state decode latency.

## Experiment reporting

Always record:

- commit hash;
- GPU model and memory;
- PyTorch, CUDA, and Transformers versions;
- model and adapter identifiers;
- prompt and output token counts;
- warm-up count and measured repetitions;
- decoder and draft length;
- TTFT, TPOT, latency, tokens/s, peak memory, and speculative acceptance;
- exact-token equality with target greedy decoding.
