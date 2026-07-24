# Grounded Qwen runtime

## Purpose

The Terraria catalog and guide corpus remain the factual grounding layer. They
are not replaced by a model-specific chatbot implementation. The LLM is added
only after routing and retrieval:

```text
question
  -> IntentRouter / EntityResolver
  -> FactService or GuideDocumentStore
  -> ContextBuilder with [S1], [S2], ... provenance
  -> QwenPairRuntime
  -> grounding validator
  -> LLM answer or deterministic fallback
```

The deterministic renderer is retained as a safety baseline. Missing,
ambiguous, failed, or uncited model output falls back to that answer.

## Model pair

The default research pair is:

- draft: `Qwen/Qwen3-0.6B`
- target: `Qwen/Qwen3-4B`

They are text-only dense Qwen3 causal language models. The runtime checks the
full tokenizer vocabulary, added tokens, and special token IDs before paired
speculative generation. This check is mandatory; sharing a brand name alone is
not sufficient.

Qwen3.5 was not selected for the first speculative-decoding experiment because
its multimodal hybrid DeltaNet/attention architecture and state cache would
require a different loader and cache-management implementation. That would mix
an architecture port with the first decoding optimization experiment.

## Existing code reused

The integration extends rather than replaces the project modules:

- `src/models/loader.py` remains the single model loader.
- `src/inference/autoregressive.py` remains the target and draft baseline.
- `src/inference/speculative.py` remains the correctness-first speculative
  baseline.
- `src/training/train_sft.py` remains the optional QLoRA training path.
- `src/assistant/*` remains the retrieval and grounding pipeline.

New orchestration lives in `src/inference/chat_runtime.py` and
`src/assistant/qwen_generator.py`.

## No training is required for the first run

The assistant can run directly with the two post-trained base checkpoints. A
LoRA adapter is optional. If an adapter is configured for the target, it must
have been trained from the exact target base checkpoint in the pair. An adapter
trained from `Qwen3-4B-Instruct-2507` must not be attached to `Qwen3-4B`.

## Commands

Build the factual databases first:

```bash
python scripts/build_terraria_knowledge.py --quiet
python scripts/build_terraria_guides.py --offline
```

Run the large-model autoregressive assistant:

```bash
python scripts/chat_terraria_llm.py \
  "进入困难模式后该做什么？" \
  --engine target \
  --debug
```

Run the small model independently:

```bash
python scripts/chat_terraria_llm.py \
  "夜之刃怎么合成？" \
  --engine draft \
  --debug
```

Run the existing correctness-first speculative baseline:

```bash
python scripts/chat_terraria_llm.py \
  "What should I do after entering Hardmode?" \
  --engine speculative \
  --debug
```

Validate and compare the model pair:

```bash
python scripts/smoke_qwen_pair.py \
  --engines draft target speculative
```

## Next inference work

The current speculative implementation is a correctness baseline. It recomputes
the draft prefix each round and is not expected to be fast. The next isolated
optimization should reuse the draft KV cache, then benchmark target-only versus
speculative decoding on identical prompts and outputs using TTFT, TPOT,
throughput, acceptance rate, forward-call counts, and peak GPU memory.

## Optional domain adapters

Training is not part of the initial inference integration. If domain adapters
are later useful, the repository includes matching configs for both members of
the pair:

```bash
python -m src.training.train_sft \
  --config configs/terraria_qwen3_0_6b_qlora.yaml

python -m src.training.train_sft \
  --config configs/terraria_qwen3_4b_qlora.yaml
```

Training both checkpoints on aligned data can improve draft/target agreement,
but acceptance-rate changes must be measured rather than assumed. The factual
database remains the source of truth even after SFT.
