# TinyQwenDraft: Custom Speculative Draft Model

## Purpose

The original character-level TinyGPT remains a compact language-modeling
exercise, but it cannot serve as a draft for Qwen because its character
vocabulary, token IDs, forward interface, positional state, and generation
cache are incompatible with the target model.

`TinyQwenDraft` gives the from-scratch model a real role in the final system:
it proposes tokens that are verified by the fixed Qwen target during greedy
speculative decoding. The target still determines the exact output. A weak
draft lowers acceptance or causes a slowdown; it does not replace the target's
answer.

The project keeps two draft tracks:

1. **Reliable baseline:** `Qwen/Qwen3-0.6B -> Qwen/Qwen3-4B`.
2. **Custom research draft:** `TinyQwenDraft -> Qwen/Qwen3-4B`.

The custom draft should be compared against the 0.6B baseline rather than
presented as guaranteed to be faster.

## Architecture

The implementation is in `src/models/tiny_qwen_draft/` and is written directly
in PyTorch. It uses:

- the exact target tokenizer and token-ID mapping;
- tied input embeddings and output projection;
- pre-RMSNorm decoder blocks;
- grouped-query attention;
- per-head query/key RMSNorm;
- rotary position embeddings;
- SwiGLU feed-forward blocks;
- bias-free projections;
- a crop-able persistent KV cache;
- a causal-LM interface with `input_ids`, `past_key_values`, `use_cache`,
  `labels`, `logits`, and `past_key_values` outputs.

The default configuration is intentionally small:

```yaml
hidden_size: 256
intermediate_size: 768
num_hidden_layers: 6
num_attention_heads: 4
num_key_value_heads: 2
max_position_embeddings: 4096
```

The full parameter count is determined after loading the target tokenizer. Most
parameters are in the shared embedding/output matrix because the target token
vocabulary is large. Weight tying prevents this matrix from being duplicated.

## Tokenizer contract

A standard speculative decoder consumes draft token IDs directly in the target.
For that reason, the custom draft must use the same:

- token-to-ID vocabulary;
- added tokens;
- BOS/EOS/PAD/UNK IDs;
- chat template used to construct the deployment prompt.

At training time, the project records SHA-256 fingerprints for the vocabulary
mapping and chat template in `config.json`. At load time,
`validate_model_tokenizer_contract()` rejects an incompatible tokenizer. The
paired runtime then performs its existing draft/target vocabulary equality
check as a second guard.

Do not train this model with the legacy `CharTokenizer` or a separately trained
BPE tokenizer.

## Persistent speculative cache

`src/inference/speculative.py` now prefills both draft and target exactly once.
For every verification round it:

1. proposes a token block from the persistent draft cache;
2. verifies it with either the target-consistent `exact` path or the one-call
   performance-oriented `block` path;
3. crops both caches to the accepted prefix after a mismatch;
4. feeds the target correction token to both models; or
5. consumes the fully accepted proposal and target bonus token in both caches.

This removes the previous behavior where the draft repeatedly reprocessed the
entire prompt at the beginning of every round. The decoder reports separate
draft and target prefill times and measures TTFT at the first actually available
output token.

The same cache-aware path is also used by model-pair analysis: the prompt is
prefilled once, and vocabulary logits are materialized only for completion
positions.

The current implementation is deliberately limited to:

- batch size 1;
- greedy decoding;
- guaranteed target-only equality in `exact` mode;
- measured, not assumed, equality in `block` mode.

Sampling-based speculative decoding requires acceptance/rejection probabilities
and residual sampling and is outside this implementation.

## Training objective

The custom draft is not trained to match a human reference answer in the
abstract. It is trained to imitate the exact continuation style of the fixed
target configuration.

Use the real grounded prompt:

```text
system instruction
+ selected structured facts
+ selected guide chunks
+ player state
+ user question
```

Then generate a target answer with the fixed Qwen target and keep only answers
that pass the grounding validator. `scripts/generate_teacher_answers.py`
already builds these chat records. Prompt tokens are masked with `-100`; only
the target-generated assistant continuation is supervised.

The custom trainer additionally uses `loss_only=True`, which projects only
hidden states that predict supervised assistant tokens. This avoids allocating
a full `[batch, sequence, vocabulary]` logits tensor for masked prompt tokens.

This is **sequence-level target adaptation**, not logits distillation. A future
KL-based trainer should remain a separate experiment.

## Colab workflow

Install dependencies:

```bash
pip install -r requirements-training.txt
```

Generate target-teacher training examples from reviewed training questions.
The input may use either the formal annotation shape with a top-level `question`
field or the cleaned chat-SFT shape with `domain`/`game` plus `messages`; in the
latter case, the generator extracts the final user message and ignores the
existing assistant answer:

```bash
python scripts/generate_teacher_answers.py \
  --input <reviewed-train-jsonl> \
  --output data/gameguide/target_teacher_train.jsonl \
  --split train \
  --config configs/gameguidelm_qwen3_pair.yaml
```

Generate a separate validation file from held-out validation questions:

```bash
python scripts/generate_teacher_answers.py \
  --input <reviewed-validation-jsonl> \
  --output data/gameguide/target_teacher_validation.jsonl \
  --split validation \
  --config configs/gameguidelm_qwen3_pair.yaml
```

Train the custom draft:

```bash
python scripts/train_tiny_qwen_draft.py \
  --config configs/tiny_qwen_draft.yaml
```

The final checkpoint is written to:

```text
results/tiny_qwen_draft/final/
```

It contains:

```text
config.json
pytorch_model.bin
tokenizer files
```

The trainer rejects records whose declared split does not match the configured
train or validation role, and it refuses paths under a formal `evaluation/`
directory. `training_report.json` records SHA-256 hashes for both input files,
target-source distributions, the seed, architecture, tokenizer fingerprints,
and the Python/PyTorch/CUDA environment.

Run the grounded assistant, alignment analyzer, or benchmark with:

```text
configs/gameguidelm_tiny_qwen_pair.yaml
```

First validate the pair and exact greedy output:

```bash
python scripts/smoke_qwen_pair.py \
  --config configs/gameguidelm_tiny_qwen_pair.yaml \
  --prompt "Explain why persistent KV cache matters." \
  --max-new-tokens 32 \
  --engines draft target speculative
```

Analyze target-token distributions:

```bash
python scripts/analyze_qwen_pair.py \
  --config configs/gameguidelm_tiny_qwen_pair.yaml \
  --game terraria \
  "What should I do after entering Hardmode?"
```

Benchmark:

```bash
python scripts/benchmark_gameguidelm.py \
  --config configs/gameguidelm_tiny_qwen_pair.yaml \
  --input data/terraria/terraria_eval.jsonl \
  --output results/tiny_draft/runs.jsonl \
  --summary results/tiny_draft/summary.json \
  --engines target speculative \
  --warmup-runs 1 \
  --runs 5
```

Use the same input files, output limit, draft length, warm-up count, and run
count when comparing this result with the pretrained Qwen3-0.6B pair.

## Required experiment matrix

Measure all three systems on identical grounded prompts:

```text
A. Qwen3-4B target-only greedy
B. Qwen3-0.6B -> Qwen3-4B greedy speculative
C. TinyQwenDraft -> Qwen3-4B greedy speculative
```

For the custom draft, test at least:

```text
draft tokens per round: 2, 4, 6, 8
prompt lengths:          256, 512, 1024, 2048
```

Report:

- parameter count and model memory;
- draft and target prefill time;
- true TTFT;
- TPOT and end-to-end latency;
- tokens per second;
- proposed and accepted tokens;
- acceptance rate;
- accepted tokens per round;
- draft and target forward calls;
- target-only/speculative output equality;
- peak GPU memory;
- speedup or slowdown relative to target-only.

The key result is not training loss. It is whether draft alignment is high
enough that saved target work exceeds the custom draft's overhead.

## Scope boundaries

- Do not remove the Qwen3-0.6B baseline.
- Do not claim the random-initialized custom draft is useful before measuring
  acceptance and latency.
- Do not train on the formal Stardew or Terraria evaluation files.
- Do not commit large model checkpoints to Git.
- Do not change the target checkpoint after teacher data generation without
  regenerating the teacher data and tokenizer contract.
- Keep Shakespeare TinyGPT only as a legacy educational smoke example; it is no
  longer the main small-model contribution.
