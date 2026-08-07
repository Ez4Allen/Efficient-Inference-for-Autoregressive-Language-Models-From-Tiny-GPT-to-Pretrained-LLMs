# Grounded target/draft runtime

## Purpose

The Terraria and Stardew knowledge layers remain the factual grounding source.
They are not replaced by a model-specific chatbot implementation. The LLM runs
only after routing and retrieval:

```text
question
  -> game IntentRouter / EntityResolver
  -> FactService or GuideStore
  -> bounded evidence with [S1], [S2], ... provenance
  -> QwenPairRuntime
  -> grounding validator
  -> LLM answer or deterministic fallback
```

The deterministic renderer is retained as the safety baseline. Missing,
ambiguous, failed, or unsupported model output falls back to that answer.

## Model pairs

The fixed target is:

```text
Qwen/Qwen3-4B
```

The runtime supports two draft tracks:

```text
Qwen/Qwen3-0.6B -> Qwen/Qwen3-4B
TinyQwenDraft   -> Qwen/Qwen3-4B
```

The pretrained 0.6B model is the reliable baseline. The custom draft is a
research model implemented in this repository and trained on fixed-target
continuations.

Paired speculative generation requires identical token IDs. The runtime checks:

- complete token-to-ID vocabulary;
- added tokens;
- BOS/EOS/PAD/UNK IDs;
- for custom checkpoints, recorded vocabulary and chat-template SHA-256.

Sharing a model-family name is not enough.

## Existing code reused

- `src/models/loader.py`: shared Hugging Face/custom checkpoint loader;
- `src/inference/autoregressive.py`: target and draft greedy baselines;
- `src/inference/speculative.py`: persistent-cache greedy speculative decoder;
- `src/inference/chat_runtime.py`: paired chat orchestration and metrics;
- `src/training/train_sft.py`: target and pretrained-draft PEFT training;
- `src/training/tiny_qwen_draft.py`: custom from-scratch draft training;
- `src/gameguide/*`: retrieval, prompting, validation, and fallback.

## Persistent cache behavior

Both models prefill the prompt exactly once. Each speculative round proposes a
block from the draft cache and verifies it with the target cache. On mismatch,
both caches are cropped to the accepted prefix before the target correction
token is processed. On full acceptance, the target bonus token is synchronized
into both models.

The current decoder supports:

- batch size one;
- greedy generation;
- exact equality with target-only greedy output.

It does not implement speculative sampling.

## Commands

Build factual backends first:

```bash
python scripts/build_terraria_knowledge.py --quiet
python scripts/build_stardew_knowledge.py --quiet
```

Target-only grounded generation:

```bash
python scripts/chat_gameguide.py \
  --game terraria \
  --llm \
  --engine target \
  "What should I do after entering Hardmode?"
```

Pretrained draft/speculative baseline uses:

```text
configs/gameguidelm_qwen3_pair.yaml
```

Custom draft/speculative research uses:

```text
configs/gameguidelm_tiny_qwen_pair.yaml
```

The custom config is valid only after `results/tiny_qwen_draft/final` has been
created.

## Training order

1. Record target-only and pretrained speculative baselines.
2. Freeze target checkpoint, tokenizer, chat template, and prompt policy.
3. Generate separate target-teacher train and validation data.
4. Optionally adapt Qwen3-0.6B with LoRA.
5. Train `TinyQwenDraft` from random initialization.
6. Validate exact tokenizer contracts.
7. Compare both drafts under identical prompts and settings.

Formal evaluation data must never be used to generate teacher training records.

## Claim boundary

The implementation now has persistent caches, but that alone does not guarantee
speedup. Acceptance, vocabulary-projection cost, target verification cost,
prompt length, output length, and GPU behavior must be measured end to end.
