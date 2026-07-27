# GameGuideLM Model Study Protocol

This document defines the model-centric experiments for the course project. The
knowledge stores are held fixed and act as a reproducible workload; the research
variables are the language-model prompt, adapter, draft/target pair, and decoder.

## 1. Research questions

1. Does retrieved evidence reduce unsupported game-guide claims across Terraria
   and Stardew Valley?
2. Does evidence-aware LoRA improve citation adherence, refusal behavior, and
   concise synthesis when retrieval is unchanged?
3. How closely does Qwen3-0.6B approximate Qwen3-4B on the same grounded
   completions?
4. When does speculative decoding reduce target-model work without changing the
   greedy Qwen3-4B output?
5. Does sequence-level teacher adaptation of the draft increase token agreement
   and end-to-end speed?
6. How do evidence budget, game, language, query type, and prompt length affect
   answer quality and model-pair alignment?

## 2. Controlled systems

| ID | Evidence | Model | Decoder | Adapter |
|---|---|---|---|---|
| D | yes | deterministic renderer | none | none |
| U0 | no | Qwen3-4B | autoregressive | none |
| T0 | yes | Qwen3-4B | autoregressive | none |
| T1 | yes | Qwen3-4B | autoregressive | evidence-aware LoRA |
| d0 | yes | Qwen3-0.6B | autoregressive | none |
| S0 | yes | 0.6B → 4B | speculative | none |
| S1 | yes | teacher-adapted 0.6B → fixed 4B | speculative | draft LoRA |

The deterministic system is the safety/reference baseline. U0 is an explicit
hallucination ablation and is not a deployed assistant.

## 3. Prompt and evidence controls

GameGuideLM applies a deterministic evidence budget before model generation:

```text
max sources
max characters per source
max total evidence characters
max structured-fact characters
```

The selected source IDs remain stable, and the validator accepts citations only
to sources that were actually placed in the prompt. Long guide chunks are
truncated at sentence or word boundaries. Large structured objects fall back to
the verified deterministic summary instead of malformed character-truncated
JSON.

Required evidence-budget ablations:

```text
B1: 2 sources / 2k evidence characters
B2: 4 sources / 4k evidence characters
B3: 6 sources / 7.2k evidence characters (default)
B4: full available evidence, subject to model context limit
```

Report prompt tokens as a measured variable. Do not compare systems that receive
different evidence without identifying the difference.

## 4. Grounded generation reliability

The first Qwen answer is checked for:

- valid source IDs;
- citations when evidence exists;
- URLs limited to retrieved sources;
- high-risk numeric claims present in the prompt support package;
- no `<think>` trace;
- non-empty and bounded output.

One constrained repair pass is allowed. The repair prompt includes the failed
answer, validation issues, and the same bounded evidence package. If the second
answer fails, the system returns the deterministic answer. Record first-pass
failure, repair success, and final fallback separately.

## 5. Warm model benchmark

Run target, draft, and speculative engines in one process so checkpoints are
loaded once. Each engine receives explicit warm-up runs before measurement.

```bash
python scripts/benchmark_gameguidelm.py \
  --input \
    data/terraria/terraria_eval.jsonl \
    data/stardew/evaluation/stardew_eval_v1.jsonl \
  --output results/model_study/runs.jsonl \
  --summary results/model_study/summary.json \
  --engines target draft speculative \
  --warmup-runs 1 \
  --runs 5 \
  --max-new-tokens 128
```

The benchmark records per-run output hashes, grounding validation, prompt and
output tokens, TTFT, TPOT, latency, tokens/s, forward calls, and speculative
acceptance. Speculative output is compared against the measured target output.

A reported speedup must use:

```text
mean target warm latency / mean speculative warm latency
```

Model download, checkpoint loading, and first CUDA initialization are reported
separately and must not be included in steady-state latency.

## 6. Model-pair distribution analysis

`analyze_qwen_pair.py` evaluates only completion logits, using prompt KV caches
instead of materializing vocabulary logits for the entire long prompt. This is
important for Qwen's 151,669-token vocabulary.

Metrics:

- top-1 agreement;
- top-k overlap;
- draft and target entropy;
- Jensen–Shannon divergence;
- log probability of the target token under each model.

Slice results by:

- Terraria versus Stardew Valley;
- structured fact versus guide question;
- English versus Chinese;
- short versus long evidence context;
- ordinary language versus names, numbers, dates, and citations.

## 7. LoRA sequence

Do not train before recording T0, d0, and S0.

### 7.1 Target evidence-aware LoRA

Construct the training prompt by executing the real retrieval pipeline. Prefer
verified deterministic targets or already validated cited references. The data
builder records whether a citation was inserted during a legacy migration; it
does not silently treat every uncited answer as grounded.

Measure T1 against T0 with retrieval held fixed.

### 7.2 Draft teacher adaptation

Freeze the final target configuration. Generate target answers that pass the
grounding validator, then train Qwen3-0.6B on the same prompts and teacher
completions. Measure S1 against S0 by token agreement, JS divergence,
acceptance, and latency—not only training loss.

This is sequence-level adaptation. True logits distillation is future work and
requires a separate KL-based trainer.

## 8. Evaluation and reporting

Model-quality metrics:

- expected-status accuracy;
- intent accuracy;
- required-fact coverage;
- forbidden-error rate;
- valid-citation rate;
- false-premise refusal correctness;
- human factuality, completeness, and usefulness.

Runtime metrics:

- TTFT;
- mean TPOT;
- end-to-end latency;
- output tokens/s;
- target and draft forward calls;
- proposed and accepted draft tokens;
- acceptance rate and accepted tokens per round;
- peak GPU memory.

Every table must state:

- commit hash;
- GPU and memory;
- PyTorch/CUDA/Transformers versions;
- exact base model and adapter IDs;
- dtype and quantization;
- warm-up and measured repetitions;
- prompt/output token counts;
- decoder and draft length;
- evidence-budget configuration.

## 9. Claim boundaries

The v1 release proves software correctness and reproducible workload
construction. It does not claim that the current correctness-first speculative
implementation is faster. A speed claim requires warm repeated measurements and
end-to-end cost accounting. Likewise, an included LoRA configuration is not a
trained model result until the adapter is trained and evaluated.
