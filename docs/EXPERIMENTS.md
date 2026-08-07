# Experimental Plan

## Research questions

1. Does evidence grounding reduce false-premise hallucination and improve
   required-fact coverage across Terraria and Stardew Valley?
2. Does evidence-aware LoRA improve citation adherence and answer organization
   beyond the base Qwen3-4B target?
3. How do Qwen3-0.6B and Qwen3-4B differ token by token on grounded prompts?
4. Can Qwen3-0.6B reduce Qwen3-4B decoding work while preserving exact greedy
   target output?
5. Can a custom Qwen-token-compatible model trained from scratch serve as a
   lower-cost speculative draft?
6. Does target-teacher sequence adaptation improve token agreement, acceptance,
   and end-to-end latency for either draft?

## Systems

| ID | Retrieval | Generator / pair | Decoder |
|---|---|---|---|
| D | yes | deterministic renderer | none |
| U0 | no | base Qwen3-4B | autoregressive |
| T0 | yes | base Qwen3-4B | autoregressive |
| T1 | yes | evidence-aware Qwen3-4B LoRA | autoregressive |
| d0 | yes | base Qwen3-0.6B | autoregressive |
| S0 | yes | base Qwen3-0.6B -> fixed Qwen3-4B | speculative |
| S1 | yes | teacher-LoRA Qwen3-0.6B -> fixed Qwen3-4B | speculative |
| c0 | yes | trained TinyQwenDraft | autoregressive |
| S2 | yes | trained TinyQwenDraft -> fixed Qwen3-4B | speculative |

U0 is a hallucination ablation, not the deployed assistant. S0 is the reliable
speculative baseline. S2 is a research comparison and may be slower.

## Evaluation slices

- Game: Terraria, Stardew Valley;
- Language: English, Chinese;
- Query: structured fact, conditional fact, guide/progression;
- Safety: false premise, ambiguity, missing player state;
- Evidence: structured only, guide only, compact hybrid, full hybrid;
- Prompt length: 256, 512, 1024, 2048 token buckets;
- Output limit: 32, 64, 128 tokens;
- Draft length: 2, 4, 6, 8 tokens.

## Model-quality metrics

- expected-status accuracy;
- intent accuracy;
- required-fact coverage;
- forbidden-error rate;
- valid-citation rate;
- refusal correctness;
- repair/fallback rate;
- human factuality and usefulness review.

## Model-pair metrics

- top-1 token agreement;
- top-k overlap;
- draft/target entropy;
- Jensen-Shannon divergence;
- target-token log probability under each model;
- proposed and accepted token counts;
- acceptance rate;
- mean accepted draft tokens per round;
- exact-token equality with target-only greedy decoding.

## Runtime metrics

- draft prefill time;
- target prefill time;
- actual time to first available output token;
- mean TPOT;
- end-to-end generation latency;
- output tokens per second;
- target and draft forward calls;
- model and KV-cache memory;
- peak GPU memory.

Model loading, download, and first-use CUDA initialization must be reported
separately from warm inference latency. Every performance result should include
warm-up count, repetitions, GPU, dtype/quantization, prompt tokens, output
tokens, target checkpoint, tokenizer fingerprint, draft checkpoint, and draft
length.

## Required ablations

1. grounded versus ungrounded target generation;
2. base target versus evidence-aware target LoRA;
3. target-only versus Qwen3-0.6B speculative;
4. target-only versus TinyQwenDraft speculative;
5. Qwen3-0.6B versus TinyQwenDraft under the same fixed target;
6. draft lengths 2/4/6/8;
7. base versus teacher-adapted Qwen3-0.6B;
8. prompt-length and output-length buckets;
9. Terraria versus Stardew;
10. structured-fact versus guide prompts.

## Execution sequence

1. Freeze the fixed target and tokenizer.
2. Record D, U0, T0, d0, and S0.
3. Generate target-teacher train and validation data from reviewed non-eval
   annotations.
4. Train the optional Qwen3-0.6B adapter.
5. Train `TinyQwenDraft` from random initialization.
6. Validate tokenizer contracts and exact greedy output equality.
7. Run alignment analysis.
8. Run warm repeated benchmarks for each pair in separate processes.
9. Run quality evaluation with retrieval and target held fixed.
10. Report both speedups and slowdowns.

## Interpretation boundary

A higher acceptance rate does not prove lower latency. Draft cost, vocabulary
projection cost, cache behavior, prompt length, target verification cost, and
adapter overhead must be measured end to end.

A lower training loss does not prove that the custom draft is useful. The final
question is whether it reduces target work enough to compensate for its own
runtime and memory cost.
