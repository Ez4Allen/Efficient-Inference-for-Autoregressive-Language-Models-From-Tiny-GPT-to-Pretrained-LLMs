# Reproducibility

## Offline software validation

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python scripts/validate_release.py --skip-pytest
```

GameGuideLM v1.1.0 has 171 offline tests. They cover the original project,
the completed Stardew release contract, SFT cleanup, bilingual deterministic
behavior, guide retrieval, the custom draft configuration and model, answer-only
training, tokenizer compatibility, loader integration, and persistent speculative
cache handling. The release metadata files at the repository root describe this
same v1.1.0 snapshot.

`validate_release.py` compiles the source, rebuilds the Terraria and Stardew
structured stores from tracked snapshots, and runs deterministic smoke queries.
It does not access the network or download model weights.

## Guide corpora

Wiki guide corpora are intentionally not committed. Build them locally:

```bash
python scripts/build_terraria_guides.py
python scripts/build_stardew_guides.py
```

After the initial online build, use `--offline` to rebuild indexes from local
raw snapshots.

## GPU model dependencies

```bash
pip install -r requirements-training.txt
```

The fixed target is:

```text
Qwen/Qwen3-4B
```

The two draft tracks are:

```text
Reliable baseline: Qwen/Qwen3-0.6B
Custom research:  results/tiny_qwen_draft/final
```

Both pair configs use the exact target tokenizer. Run the pretrained pair smoke
test before collecting results:

```bash
python scripts/smoke_qwen_pair.py \
  --config configs/gameguidelm_qwen3_pair.yaml \
  --prompt "Explain why draft and target tokenizers must match." \
  --max-new-tokens 32 \
  --engines draft target speculative
```

## Custom draft data and training

Generate train and validation teacher data from separate reviewed non-evaluation
annotations:

```bash
python scripts/generate_teacher_answers.py \
  --input <reviewed-training-jsonl> \
  --output data/gameguide/target_teacher_train.jsonl \
  --split train \
  --config configs/gameguidelm_qwen3_pair.yaml

python scripts/generate_teacher_answers.py \
  --input <reviewed-validation-jsonl> \
  --output data/gameguide/target_teacher_validation.jsonl \
  --split validation \
  --config configs/gameguidelm_qwen3_pair.yaml
```

Train:

```bash
python scripts/train_tiny_qwen_draft.py \
  --config configs/tiny_qwen_draft.yaml
```

The checkpoint records:

- architecture configuration;
- target and tokenizer references;
- exact tokenizer vocabulary SHA-256;
- chat-template SHA-256;
- special token IDs;
- training configuration and history.

Do not commit the generated checkpoint to Git.

## Custom pair validation

After training:

```bash
python scripts/smoke_qwen_pair.py \
  --config configs/gameguidelm_tiny_qwen_pair.yaml \
  --prompt "Explain why persistent KV cache matters." \
  --max-new-tokens 32 \
  --engines draft target speculative
```

The speculative text must exactly match target-only greedy text.

## Warm benchmarking

Run each pair in a separate process under identical settings:

```bash
python scripts/benchmark_gameguidelm.py \
  --config configs/gameguidelm_qwen3_pair.yaml \
  --input data/terraria/terraria_eval.jsonl \
  --output results/qwen_pair/runs.jsonl \
  --summary results/qwen_pair/summary.json \
  --engines target draft speculative \
  --warmup-runs 1 \
  --runs 5

python scripts/benchmark_gameguidelm.py \
  --config configs/gameguidelm_tiny_qwen_pair.yaml \
  --input data/terraria/terraria_eval.jsonl \
  --output results/tiny_pair/runs.jsonl \
  --summary results/tiny_pair/summary.json \
  --engines target draft speculative \
  --warmup-runs 1 \
  --runs 5
```

Checkpoint download, loading, and first CUDA initialization must not be mixed
into steady-state decode latency.

## Experiment reporting

Always record:

- commit hash;
- GPU model and memory;
- PyTorch, CUDA, and Transformers versions;
- model, tokenizer, and adapter identifiers;
- custom-draft parameter count and checkpoint hash;
- vocabulary/chat-template fingerprints;
- teacher train/validation dataset hashes;
- random seed;
- prompt and output token counts;
- warm-up count and measured repetitions;
- decoder and draft length;
- draft/target prefill, TTFT, TPOT, latency, tokens/s, and peak memory;
- proposed and accepted tokens, acceptance, and forward calls;
- exact-token equality with target-only greedy decoding.
