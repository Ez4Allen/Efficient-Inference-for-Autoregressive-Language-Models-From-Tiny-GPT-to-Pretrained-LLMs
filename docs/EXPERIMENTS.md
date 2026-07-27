# Experimental Plan

## Research questions

1. Does evidence grounding reduce false-premise hallucination and improve
   required-fact coverage across two games?
2. Does evidence-aware LoRA improve citation adherence and answer organization
   beyond the base Qwen3-4B model?
3. How do Qwen3-0.6B and Qwen3-4B differ token by token on grounded game-guide
   prompts?
4. Can the 0.6B model reduce 4B decoding work through speculative decoding
   while preserving the exact greedy target output?
5. Does sequence-level draft adaptation improve token agreement and
   speculative acceptance rate?

## Systems

| ID | Retrieval | Generator | Decoder |
|---|---|---|---|
| D | yes | deterministic renderer | none |
| T0 | yes | base Qwen3-4B | autoregressive |
| T1 | yes | evidence-aware Qwen3-4B LoRA | autoregressive |
| S0 | yes | base 0.6B → base 4B | speculative |
| S1 | yes | teacher-LoRA 0.6B → fixed 4B | speculative |

An optional ungrounded Qwen baseline may be included only for hallucination
ablation; it must not be presented as the deployed assistant.

## Evaluation slices

- Game: Terraria, Stardew Valley;
- Language: English, Chinese;
- Query: structured fact, conditional fact, guide/progression;
- Safety: false premise, ambiguity, missing player state;
- Context: short fact evidence, long guide evidence;
- Generation length and prompt length buckets.

## Model-quality metrics

- expected-status accuracy;
- intent accuracy;
- required-fact coverage;
- forbidden-error rate;
- valid-citation rate;
- refusal correctness;
- human factuality and usefulness review.

## Model-pair metrics

- top-1 token agreement;
- top-k overlap;
- draft/target entropy;
- Jensen-Shannon divergence;
- target-token log probability under each model;
- speculative acceptance rate;
- mean accepted draft tokens per round;
- exact-token equality with target greedy decoding.

## Runtime metrics

- TTFT;
- mean TPOT;
- end-to-end generation latency;
- output tokens per second;
- target and draft forward calls;
- peak GPU memory.

Model loading and first-download time must be reported separately from warm
inference latency. Each performance result should include warm-up, repeated
runs, hardware, dtype/quantization, prompt tokens, output tokens, and decoder
configuration.

## Required ablations

1. Grounded versus ungrounded target generation;
2. Base target versus target LoRA;
3. Target autoregressive versus training-free speculative;
4. Draft length 1/2/4/6/8;
5. Base draft versus teacher-adapted draft;
6. Terraria versus Stardew and fact versus guide prompts.

## Interpretation boundary

A higher acceptance rate does not by itself prove lower latency. Draft cost,
cache behavior, prompt length, verification cost, and adapter overhead must be
measured end to end.
