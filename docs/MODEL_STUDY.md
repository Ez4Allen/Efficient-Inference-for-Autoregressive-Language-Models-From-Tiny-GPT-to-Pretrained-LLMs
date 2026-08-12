# GameGuideLM Model Study Protocol

> **Current custom-model plan.** The frozen Qwen3-4B RAG and speculative
> results remain part of the report.  In response to presentation feedback, the
> new controlled custom-model study aligns the fixed 43.5M architecture with
> Qwen3-0.6B and compares scratch distillation, project-local pretraining plus
> distillation, and grounded game adaptation.  See
> `docs/CUSTOM_MODEL_STUDY.md`.

This document defines the model-centric experiments for the course project. The
knowledge stores are held fixed and act as a reproducible workload. Research
variables are evidence, target adaptation, draft model, draft adaptation, and
decoder.

## 1. Research questions

1. Does retrieved evidence reduce unsupported game-guide claims across Terraria
   and Stardew Valley?
2. Does evidence-aware target LoRA improve citation adherence, refusal behavior,
   and concise synthesis when retrieval is unchanged?
3. How closely does a pretrained Qwen3-0.6B draft approximate the fixed
   Qwen3-4B target on grounded completions?
4. Can a custom Qwen-token-compatible draft trained from scratch provide a
   useful cost/acceptance trade-off?
5. When does speculative decoding reduce target-model work without changing the
   greedy target output?
6. Does target-teacher sequence adaptation improve token agreement, acceptance,
   and end-to-end latency?
7. How do game, language, question type, evidence budget, prompt length, and
   output length affect model-pair alignment?

## 2. Controlled systems

| ID | Evidence | Draft / target | Decoder | Training |
|---|---|---|---|---|
| D | yes | deterministic renderer | none | none |
| U0 | no | Qwen3-4B | autoregressive | none |
| T0 | yes | Qwen3-4B | autoregressive | none |
| T1 | yes | Qwen3-4B | autoregressive | evidence-aware LoRA |
| d0 | yes | Qwen3-0.6B | autoregressive | none |
| S0 | yes | Qwen3-0.6B -> Qwen3-4B | speculative | none |
| S1 | yes | adapted Qwen3-0.6B -> fixed Qwen3-4B | speculative | teacher-answer LoRA |
| c0 | yes | TinyQwenDraft | autoregressive | target-teacher sequence adaptation |
| S2 | yes | TinyQwenDraft -> fixed Qwen3-4B | speculative | target-teacher sequence adaptation |

D is the safety/reference baseline. U0 is an explicit hallucination ablation and
is not a deployed assistant. S0 is the reliable speculative baseline. S2 is a
research comparison and must not be presented as guaranteed to be faster.

## 3. Prompt and evidence controls

GameGuideLM applies deterministic limits before model generation:

```text
maximum evidence sources
maximum total evidence characters
structured versus guide evidence policy
prompt mode
generation limit
```

Selected source IDs remain stable, and citations are accepted only for sources
that actually appear in the prompt. Long guide chunks are trimmed at natural
boundaries. Large structured objects use verified deterministic summaries rather
than malformed character-truncated JSON.

Required evidence-budget ablations:

```text
B1: 2 sources / 2,000 evidence characters
B2: 4 sources / 4,000 evidence characters
B3: 6 sources / 7,200 evidence characters
B4: full available evidence, subject to both model context limits
```

Measure prompt tokens. Never compare systems that receive different evidence
without stating the difference.

## 4. Grounded generation reliability

The first target answer is checked for:

- valid source IDs;
- citations when evidence exists;
- URLs limited to retrieved sources;
- high-risk numeric claims supported by the prompt package;
- no leaked thinking trace;
- non-empty and bounded output.

One constrained repair pass is allowed. If repair also fails, the deterministic
answer is returned. Record first-pass failure, repair success, and final fallback
separately.

## 5. Tokenizer and target freeze

All speculative pairs must use the exact target token-to-ID mapping and special
IDs. `TinyQwenDraft` additionally records vocabulary and chat-template SHA-256
fingerprints in its checkpoint. Loading fails if the supplied tokenizer does not
match.

Before teacher-data generation, freeze:

```text
target checkpoint
target adapter, if any
tokenizer files
chat template
grounding prompt mode
evidence budget
generation settings
```

Changing any of these invalidates the teacher-data contract.

## 6. Warm model benchmark

Run one pair per process, load checkpoints once, warm each selected engine, then
repeat identical cases:

```bash
python scripts/benchmark_gameguidelm.py \
  --config configs/gameguidelm_qwen3_pair.yaml \
  --input \
    data/terraria/terraria_eval.jsonl \
    data/stardew/evaluation/stardew_eval_v1.jsonl \
  --output results/model_study/qwen_pair_runs.jsonl \
  --summary results/model_study/qwen_pair_summary.json \
  --engines target draft speculative \
  --warmup-runs 1 \
  --runs 5 \
  --max-new-tokens 128
```

After the custom checkpoint is trained, repeat with:

```bash
python scripts/benchmark_gameguidelm.py \
  --config configs/gameguidelm_tiny_qwen_pair.yaml \
  --input \
    data/terraria/terraria_eval.jsonl \
    data/stardew/evaluation/stardew_eval_v1.jsonl \
  --output results/model_study/tiny_pair_runs.jsonl \
  --summary results/model_study/tiny_pair_summary.json \
  --engines target draft speculative \
  --warmup-runs 1 \
  --runs 5 \
  --max-new-tokens 128
```

The decoder prefills draft and target once and maintains both KV caches across
rounds. After a mismatch, it crops both caches to the accepted prefix and feeds
the target correction token to both models. A fully accepted block is followed
by the target bonus token, which is also synchronized into both caches.

The benchmark records:

- generated token IDs, output hash, verification mode, and exact
  target/speculative match;
- prompt and output tokens;
- draft and target prefill times;
- actual time to first available output token;
- TPOT, total latency, and tokens/s;
- draft and target forward calls;
- proposed and accepted tokens;
- acceptance and accepted tokens per round;
- grounding validation and environment metadata.

A speedup must use:

```text
mean target warm latency / mean speculative warm latency
```

Download, load, and first CUDA initialization time must not be mixed into steady
state generation latency.

## 7. Draft-length and prompt-length matrix

Test at least:

```text
draft tokens per round: 2, 4, 6, 8
prompt tokens:          256, 512, 1024, 2048
output limits:           32, 64, 128
```

Use identical prompts and output limits for the pretrained and custom drafts.
A higher acceptance rate is not automatically better if the draft itself is
substantially slower.

## 8. Model-pair distribution analysis

`analyze_qwen_pair.py` evaluates the fixed target continuation under both
models. The analysis should materialize vocabulary logits only for completion
positions, not for the entire long evidence prompt.

Metrics:

- top-1 agreement;
- top-k overlap;
- draft and target entropy;
- Jensen-Shannon divergence;
- target-token log probability under each model.

Slice by:

- Terraria versus Stardew Valley;
- structured fact versus guide/progression question;
- English versus Chinese;
- short versus long evidence context;
- ordinary language versus names, numbers, dates, and citations.

## 9. Training sequence

Do not train before recording T0, d0, and S0.

### 9.1 Target evidence-aware LoRA

Construct training prompts by executing the real retrieval pipeline over
reviewed training annotations. Measure T1 against T0 with retrieval fixed.

### 9.2 Pretrained draft adaptation

Generate validated target answers, then train Qwen3-0.6B on the same prompts and
target continuations. Measure S1 against S0.

### 9.3 Custom draft adaptation

Initialize `TinyQwenDraft` from random weights, use the exact target tokenizer,
and train on the same target continuations. Measure c0 and S2. Do not use formal
evaluation questions as teacher-data input.

Both draft paths are sequence-level adaptation. True logits distillation is
future work and requires a separate KL-based trainer.

## 10. Evaluation and reporting

Model-quality metrics:

- expected-status accuracy;
- intent accuracy;
- required-fact coverage;
- forbidden-error rate;
- valid-citation rate;
- false-premise refusal correctness;
- human factuality, completeness, and usefulness.

Runtime metrics:

- draft/target prefill;
- TTFT;
- mean TPOT;
- end-to-end latency;
- output tokens/s;
- target and draft forward calls;
- proposed and accepted draft tokens;
- acceptance and accepted tokens per round;
- peak GPU memory.

Every table must state:

- commit hash;
- GPU and memory;
- PyTorch, CUDA, and Transformers versions;
- exact model, tokenizer, and adapter identifiers;
- custom-draft parameter count and checkpoint hash;
- dtype and quantization;
- warm-up and measured repetitions;
- prompt/output token counts;
- decoder and draft length;
- evidence-budget configuration;
- training and evaluation dataset hashes.

## 11. Claim boundaries

The current branch proves software correctness for the custom architecture,
cache contract, loader integration, answer-only training path, and exact greedy
speculative output. It does not yet prove that the custom draft is useful. That
claim requires an actual trained checkpoint and warm repeated GPU results.

A slowdown is a valid result and must be reported. Training loss, parameter
count, or acceptance rate alone is not an end-to-end speed result.
