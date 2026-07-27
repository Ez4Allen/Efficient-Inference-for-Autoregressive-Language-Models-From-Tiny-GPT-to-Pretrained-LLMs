# Course Project Proposal — GameGuideLM

## Title

**GameGuideLM: Evidence-Grounded Multi-Game Question Answering with Qwen, LoRA, and Draft/Target Model Analysis**

## Motivation

Game-guide questions combine rapidly changing factual knowledge with open-ended
planning and explanation. A language model may produce fluent answers while
inventing items, recipes, schedules, drop rates, or progression requirements.
Training the model to memorize an entire Wiki is difficult to update and does
not provide reliable provenance.

GameGuideLM separates factual memory from language generation. Game-specific
plug-ins retrieve structured facts and guide excerpts, and a shared Qwen model
turns the evidence into a natural answer. Deterministic services handle
calculations such as crop deadlines and conditional availability. Generated
answers are checked for valid citations, URLs, and numeric claims, with a safe
deterministic fallback.

## Research questions

1. How much does evidence conditioning improve factual coverage and reduce
   false-premise hallucination across Terraria and Stardew Valley?
2. Can evidence-aware LoRA improve citation adherence, refusal behavior, and
   answer organization without training the model to memorize game facts?
3. How closely does a Qwen3-0.6B draft match a Qwen3-4B target on grounded game
   prompts at the token-distribution level?
4. Can speculative decoding reduce target-model decoding work while preserving
   the exact greedy target output?
5. Does sequence-level draft adaptation improve token agreement and
   speculative acceptance?

## Method

### Multi-game evidence layer

- Terraria: high-coverage structured catalog plus Official Wiki guides;
- Stardew Valley: conditional facts involving season/day/weather/time/location,
  plus a separate Wiki guide pipeline;
- shared model-facing `GameGuideResult` and `GameEvidence` contract.

### Language model

- target: Qwen3-4B;
- draft: Qwen3-0.6B from the same tokenizer-compatible family;
- deterministic, target, draft, and speculative generation modes;
- optional 4B evidence-aware LoRA and 0.6B teacher-answer LoRA.

### Evaluation

- expected-status and intent accuracy;
- required-fact coverage and forbidden-error rate;
- citation validity and false-premise refusal;
- human factuality/usefulness review;
- top-1 agreement, top-k overlap, entropy, and JS divergence;
- TTFT, TPOT, end-to-end latency, tokens/s, forward calls, memory, and
  speculative acceptance.

## Expected contribution

The project contributes a model-centric experimental framework for reliable
multi-game QA, where the same evidence-conditioned prompts support grounding
experiments, LoRA training, model-pair analysis, and speculative decoding.
Knowledge databases are treated as external, versioned model memory rather than
the final research contribution.
