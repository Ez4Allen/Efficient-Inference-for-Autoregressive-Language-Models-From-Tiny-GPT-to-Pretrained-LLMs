# TinyQwenDraft Implementation Summary

## 1. What changed

The project now has two speculative-decoding draft tracks:

1. **Reliable pretrained baseline:** `Qwen/Qwen3-0.6B -> Qwen/Qwen3-4B`
2. **Custom research draft:** `TinyQwenDraft -> Qwen/Qwen3-4B`

The legacy character-level Shakespeare TinyGPT remains only as a small educational example. It is no longer the main small-model contribution in the project.

The custom draft is intended to perform a real systems role: it proposes tokens, while the Qwen3-4B target verifies them and determines the exact greedy output.

## 2. Custom model implementation

Added:

```text
src/models/tiny_qwen_draft/
├── __init__.py
├── cache.py
├── config.py
└── model.py
```

`TinyQwenDraft` is implemented directly in PyTorch and includes:

- the exact target tokenizer vocabulary and token IDs;
- tied input embeddings and language-model output weights;
- pre-RMSNorm decoder blocks;
- grouped-query attention;
- per-head query/key RMSNorm;
- rotary position embeddings;
- SwiGLU feed-forward layers;
- bias-free linear projections;
- causal attention;
- persistent, crop-able KV cache;
- incremental decoding positions;
- a Hugging-Face-like causal-LM interface;
- assistant-only language-model loss;
- checkpoint save/load support.

The default research configuration is:

```yaml
hidden_size: 256
intermediate_size: 768
num_hidden_layers: 6
num_attention_heads: 4
num_key_value_heads: 2
max_position_embeddings: 4096
```

The exact parameter count is computed after loading the target tokenizer. Most parameters are expected to be in the shared token embedding/output matrix because the Qwen vocabulary is large.

## 3. Exact tokenizer contract

Added:

```text
src/models/tokenizer_contract.py
```

The training and runtime path now records and validates:

- the full token-to-ID mapping;
- the required embedding vocabulary size;
- BOS, EOS, PAD, and UNK token IDs;
- a SHA-256 fingerprint of the vocabulary mapping;
- a SHA-256 fingerprint of the chat template.

The final custom checkpoint saves a local copy of the target tokenizer. The loader prefers that local copy, then validates it against the contract stored in `config.json`.

The paired runtime still performs a second exact draft/target tokenizer compatibility check.

## 4. Persistent speculative decoding

Reworked:

```text
src/inference/speculative.py
```

The previous implementation repeatedly reconstructed draft state across speculative rounds. The new greedy decoder:

1. prefills the draft prompt once;
2. prefills the target prompt once;
3. keeps both KV caches across rounds;
4. proposes a token block from the draft cache;
5. verifies that block with the target;
6. crops both caches after a mismatch;
7. synchronizes the target correction token into both caches;
8. synchronizes the target bonus token after a fully accepted block;
9. provides an `exact` verification path for deterministic target-token equality and a separate `block` path for speed experiments.

The current supported scope is intentionally narrow:

```text
batch size: 1
decoding: greedy only
correctness target: exact target-only output equality in `exact` mode; measured equality in `block` mode
```

Sampling-based speculative decoding is not implemented yet.

The runtime now reports:

- draft prefill time;
- target prefill time;
- actual time to first output token;
- draft and target forward-call counts;
- proposed and accepted draft tokens;
- acceptance rate;
- accepted tokens per round;
- total latency and throughput.

## 5. Training and target adaptation

Added:

```text
src/training/tiny_qwen_draft.py
scripts/train_tiny_qwen_draft.py
configs/tiny_qwen_draft.yaml
```

The custom draft is trained from random initialization using **sequence-level target adaptation**:

```text
real grounded prompt -> fixed Qwen3-4B continuation -> TinyQwenDraft supervision
```

Prompt tokens are masked. Only assistant continuation tokens contribute to the loss.

To avoid constructing a full `[batch, sequence, 150k+ vocabulary]` tensor for masked prompt positions, `loss_only=True` projects only hidden states that predict supervised assistant tokens.

The trainer includes:

- deterministic seeding;
- BF16/FP16 autocast on CUDA;
- gradient accumulation;
- gradient clipping;
- AdamW;
- warmup plus linear decay;
- periodic validation;
- periodic checkpoints;
- final tokenizer-inclusive checkpoint;
- train/validation file SHA-256 hashes;
- target-source distributions;
- architecture and tokenizer fingerprints;
- Python, PyTorch, CUDA, and GPU environment metadata.

The trainer rejects:

- records whose declared split does not match `train` or `validation`;
- `eval`/`test` records;
- inputs under a formal `evaluation/` directory.

This prevents the formal benchmark from entering draft-model adaptation.

## 6. Current Stardew SFT compatibility

`generate_teacher_answers.py` now accepts either:

1. annotation records containing a top-level `question`; or
2. cleaned chat-SFT records containing `game`/`domain` plus `messages`.

For chat-SFT records, it extracts the final user message and ignores the existing assistant answer. This means the current Stardew candidate files can be used as question sources **after cleanup and human approval**; the target model will generate the actual distillation continuation.

Do not treat the current AI-assisted Stardew candidates as approved merely because older records contain `verified=true`.

## 7. Runtime and model-loader integration

Updated:

```text
src/models/loader.py
src/models/runtime_config.py
src/inference/chat_runtime.py
```

The project can now load:

- normal Hugging Face causal LMs;
- local `TinyQwenDraft` checkpoints;
- separate model and tokenizer references;
- the pretrained Qwen draft/target pair;
- the custom draft/Qwen target pair.

New pair configuration:

```text
configs/gameguidelm_tiny_qwen_pair.yaml
```

The custom model is kept in BF16/FP16. BitsAndBytes 4-bit loading and PEFT adapters are intentionally rejected for the custom draft checkpoint.

## 8. Cache-aware draft/target analysis

Updated:

```text
src/evaluation/model_pair_alignment.py
```

The model-pair analyzer now:

- prefills the prompt once per model;
- materializes vocabulary logits only for completion positions;
- computes top-1 agreement, top-k overlap, entropy, Jensen-Shannon divergence, and target-token log probabilities without allocating prompt-wide vocabulary logits.

## 9. Validation completed in this environment

The final code passed:

```text
python -m compileall -q src scripts tests
pytest -q
```

Result:

```text
184 passed
```

The release integrity validator also passed:

```text
python scripts/validate_release.py --skip-pytest
```

Validated local knowledge data:

```text
Stardew catalog records: 505
Stardew acquisition relations: 317
Stardew guide pages/chunks: 25/100
Stardew deterministic regression: 100/100
Terraria resolved references: 14,353
```

The custom draft implementation is now integrated with the final Stardew course-release snapshot. The complete merged release was tested again:

```text
184 passed
Stardew release build: passed
release validation: engineering_passed_human_review_pending
```

## 10. Important boundary

No GPU model training was performed in this execution environment because the environment does not contain Transformers/model checkpoints and has no CUDA GPU.

Therefore, the following do **not** exist yet:

- a trained TinyQwenDraft checkpoint;
- measured draft acceptance rate against Qwen3-4B;
- measured target-call reduction;
- measured TTFT/TPOT or throughput speedup;
- proof that the custom draft is faster than Qwen3-0.6B;
- GPU memory measurements.

Those are the next Colab experiments. The implementation is ready for them, but training loss alone must not be reported as evidence that speculative decoding is useful.

## 11. Apply the rebased code patch

The new patch is based on the uploaded merged `main (4).zip`. It includes the
completed Stardew code/data release and the TinyQwenDraft integration, but it
does not include the final report or PPTX/PDF presentation.

Upload `GameGuideLM_v1.1.0_Code_Release_Rebased.patch` to Colab, then run:

```bash
cd /content/llm_project

git checkout main
git pull origin main
git checkout -b feature/gameguidelm-v1.1-code-release

git apply --check /content/GameGuideLM_v1.1.0_Code_Release_Rebased.patch
git apply /content/GameGuideLM_v1.1.0_Code_Release_Rebased.patch

python -m compileall -q src scripts tests
python -m pytest -q
python scripts/build_stardew_release.py --skip-tests
```

Expected repository test result:

```text
184 passed
```

After inspection, commit and push the feature branch. Open the Pull Request
against `main`; do not apply the patch directly on an uncommitted working tree.

## 12. Recommended Colab experiment order

### Step 1: choose the training source deliberately

The legacy 1,262 SFT candidates have been structurally cleaned, but all remain
`review_status=pending`. Do not train on them as though they were human-approved.
For the first reproducible run, use the deterministic grounded records under
`data/stardew/training/`; use legacy candidates only after real source review.
Never use the formal evaluation files for training.

### Step 2: install training dependencies

```bash
pip install -r requirements-training.txt
```

### Step 3: generate Qwen3-4B teacher continuations

For the first reproducible run, use the deterministic grounded training split:

```bash
python scripts/generate_teacher_answers.py \
  --input data/stardew/training/stardew_grounded_train_v1.jsonl \
  --output data/gameguide/target_teacher_train.jsonl \
  --split train \
  --config configs/gameguidelm_qwen3_pair.yaml
```

Generate held-out teacher validation data separately:

```bash
python scripts/generate_teacher_answers.py \
  --input data/stardew/training/stardew_grounded_validation_v1.jsonl \
  --output data/gameguide/target_teacher_validation.jsonl \
  --split validation \
  --config configs/gameguidelm_qwen3_pair.yaml
```

### Step 4: train the custom draft

```bash
python scripts/train_tiny_qwen_draft.py \
  --config configs/tiny_qwen_draft.yaml
```

Expected final checkpoint directory:

```text
results/tiny_qwen_draft/final/
```

Do not commit the checkpoint to GitHub.

### Step 5: verify exact greedy correctness

```bash
python scripts/smoke_qwen_pair.py \
  --config configs/gameguidelm_tiny_qwen_pair.yaml \
  --prompt "What should I prioritize during my first spring in Stardew Valley?" \
  --max-new-tokens 64 \
  --engines draft target speculative
```

The target and speculative generated token sequences must match exactly.

### Step 6: benchmark three systems

Use identical prompts and generation settings for:

```text
A. Qwen3-4B target only
B. Qwen3-0.6B -> Qwen3-4B speculative
C. TinyQwenDraft -> Qwen3-4B speculative
```

Test at least:

```text
draft tokens per round: 2, 4, 6, 8
prompt lengths: 256, 512, 1024, 2048
```

Report speedup even when it is below `1.0x`; a slowdown is still a valid experimental result.

## 13. Suggested PR title

```text
Add custom TinyQwen draft and persistent speculative decoding
```

## 14. Suggested PR description summary

```text
This PR replaces the Shakespeare-only TinyGPT as the main small-model research
track with a custom Qwen-token-compatible speculative draft. It adds a
Qwen-style PyTorch decoder, exact tokenizer contracts, persistent draft/target
KV caches, sequence-level target adaptation, cache-aware alignment analysis,
and benchmark-ready runtime metrics. The pretrained Qwen3-0.6B draft remains
the reliable baseline. No trained custom checkpoint or speedup claim is included
in this PR; those require the follow-up Colab GPU experiment.
```
