# GameGuideLM Architecture

## 1. Model-facing abstraction

Every game plug-in returns a `GameGuideResult`:

```text
game, status, question, intent, entity
facts, warnings, candidates
GameEvidence[S1..Sn]
deterministic answer
```

The model layer consumes only this contract. It does not contain Terraria- or
Stardew-specific SQL, aliases, or progression rules.

Relevant code:

```text
src/gameguide/schemas.py
src/gameguide/plugin.py
src/gameguide/assistant.py
src/gameguide/prompting.py
src/gameguide/generator.py
src/gameguide/validation.py
```

## 2. Game plug-ins

### Terraria

```text
TerrariaGamePlugin
  -> bilingual TerrariaAssistant
  -> TerrariaFactService for Items/NPCs/Recipes/Drops
  -> GuideDocumentStore for progression/mechanics
```

Terraria remains the high-coverage reference workload.

### Stardew Valley

```text
StardewAssistant
  -> StardewIntentRouter
  -> StardewFactService for crops/fish/villagers/recipes/bundles
  -> StardewGuideStore for guide questions
  -> StardewRenderer deterministic decision
```

The Stardew plug-in adds season, day, weather, time, location, bundle mode, and
other player-state conditions. This proves that the model interface is not
Terraria-specific.

## 3. Grounded generation

```text
GameGuideResult
       ↓
EvidenceSelectionConfig
       ↓
prepare_gameguide_prompt()
       ↓
QwenPairRuntime
       ├── Qwen3-4B target
       ├── Qwen3-0.6B pretrained draft
       ├── TinyQwenDraft custom draft
       └── target / draft / speculative engine
       ↓
validate_gameguide_answer()
       ├── valid evidence IDs
       ├── supported URLs
       ├── supported numeric claims
       ├── no thinking trace
       └── length/empty checks
       ↓
valid answer
or one constrained repair pass
or deterministic fallback
```

An explicit `UngroundedQwenGenerator` exists only for ablation. It uses the same
target checkpoint without retrieved evidence and is never the safe deployed
path.

The prompt builder keeps citation IDs stable while limiting source count,
per-source text, total evidence text, and large fact objects. The validator is
restricted to support payloads actually shown to the model; omitted evidence
cannot be cited accidentally.

## 4. Target and draft loading

`src/models/loader.py` supports two model classes through one interface:

```text
Hugging Face causal LM
local TinyQwenDraft checkpoint
```

A local custom checkpoint is identified by:

```json
{"model_type": "tiny_qwen_draft"}
```

The runtime can specify `tokenizer_name_or_path` independently of the weight
path. This is essential for the custom draft, whose weights are local while its
token contract comes from the fixed target tokenizer.

Before paired generation:

1. the custom checkpoint validates its vocabulary size, special IDs, vocabulary
   SHA-256, and chat-template SHA-256;
2. the pair validates complete draft/target vocabulary equality, added tokens,
   and special IDs;
3. both models must be on the same device for the current decoder.

## 5. TinyQwenDraft

Implementation:

```text
src/models/tiny_qwen_draft/config.py
src/models/tiny_qwen_draft/cache.py
src/models/tiny_qwen_draft/model.py
```

Architecture:

```text
target token embedding (tied to LM head)
  -> N decoder blocks
       pre-RMSNorm
       grouped-query causal self-attention
       per-head Q/K RMSNorm
       rotary position embedding
       residual
       pre-RMSNorm
       SwiGLU MLP
       residual
  -> final RMSNorm
  -> tied vocabulary projection
```

The model exposes a Hugging-Face-like causal-LM interface:

```text
input_ids
attention_mask
position_ids
past_key_values
use_cache
labels
return_dict
```

and returns:

```text
logits
loss
past_key_values
```

The cache stores one `(key, value)` pair per layer and supports in-place
`crop(sequence_length)`, which speculative mismatch recovery requires.

## 6. Persistent-cache speculative decoding

`src/inference/speculative.py` implements batch-size-one greedy speculative
decoding.

```text
Draft prefill prompt once -> draft cache + next logits
Target prefill prompt once -> target cache + next logits

repeat:
  draft proposes gamma tokens from its persistent cache
  target verifies the block in one call

  mismatch at position i:
    accept draft prefix [0:i]
    crop both caches to accepted context
    append target correction token
    feed correction into both caches

  all proposals accepted:
    append all draft tokens
    append target bonus token
    feed accepted final proposal/bonus into the required caches
```

The result is token-identical to target-only greedy decoding. The current scope
is intentionally limited to greedy generation and batch size one. Sampling
requires probability-based acceptance/rejection and residual sampling and is a
separate algorithm.

## 7. Training

### Target and pretrained draft

`src/training/train_sft.py` remains the PEFT/QLoRA trainer for:

- Qwen3-4B evidence-aware target LoRA;
- Qwen3-0.6B target-teacher draft LoRA.

### Custom draft

`src/training/tiny_qwen_draft.py` trains the custom model from random
initialization. It reuses the generic chat SFT dataset and collator but has its
own optimizer loop, checkpoint metadata, and exact tokenizer contract.

```text
reviewed training question
  -> real retrieval and evidence selection
  -> fixed target answer
  -> grounding validation
  -> target-teacher chat record
  -> answer-only TinyQwenDraft cross-entropy
```

With `loss_only=True`, only hidden states predicting supervised assistant tokens
are projected through the large vocabulary matrix. Prompt and padding tokens do
not allocate unnecessary training logits.

## 8. Model-pair research

The same grounded prompts support:

```text
Qwen3-4B target-only greedy
Qwen3-0.6B -> Qwen3-4B speculative
TinyQwenDraft -> Qwen3-4B speculative
```

`src/evaluation/model_pair_alignment.py` measures:

- top-1 agreement;
- top-k overlap;
- entropy;
- Jensen-Shannon divergence;
- target-token likelihood under draft and target.

Warm benchmarking reports draft/target prefill separately, actual TTFT, TPOT,
total latency, throughput, forward calls, acceptance, accepted tokens per round,
and exact target/speculative output equality.

## 9. Reliability boundaries

- Mutable facts remain outside model parameters.
- Deterministic services handle calculations and conditions.
- Formal evaluation data is not model-training input.
- The target determines final speculative output.
- Invalid generations fall back safely.
- False premises and missing player state are first-class evaluation cases.
- Generated Wiki corpora and model weights are built or downloaded locally.
- A trained custom draft is not useful merely because its loss decreases.
- Speed is claimed only from warm repeated end-to-end measurements.
