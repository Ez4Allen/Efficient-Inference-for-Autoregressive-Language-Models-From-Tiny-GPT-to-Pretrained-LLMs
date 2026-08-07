# GameGuideLM v1.1.0 — Code and Data Delivery

## Delivery status

```text
Offline engineering build:                 passed
Repository tests:                          184 passed
Stardew deterministic regression:          100 / 100 passed
Stardew structured records:                505
Stardew acquisition relations:             317
Stardew guide seed:                         25 pages / 100 chunks
Stardew formal regression files:            40 validation / 60 eval
Stardew grounded training data:             159 train / 17 validation
Legacy Stardew SFT audit:                   1,262 pending candidates
SQLite integrity:                           ok
Independent human benchmark approval:       pending
Qwen/TinyQwen GPU training and speed study: pending GPU execution
```

The delivery is executable, testable, and demonstration-ready without network access or model weights. It does not fabricate a human reviewer, trained checkpoint, acceptance rate, or speculative-decoding speedup.

## One-command reproduction

```bash
pip install -r requirements-dev.txt
python scripts/build_stardew_release.py
```

This regenerates data, audits the legacy SFT pool, rebuilds structured and guide databases, runs evaluation and validation, creates the static demo, and runs the full test suite.

## Main deliverables

### Software

- shared GameGuideLM plug-in and evidence contracts;
- Terraria reference implementation;
- completed Stardew structured fact, guide retrieval, routing, rendering, and evaluation module;
- Qwen target/draft runtime;
- correctness-first persistent-cache speculative decoder;
- custom Qwen-token-compatible `TinyQwenDraft` implementation and training pipeline;
- data, leakage, integrity, and release validators.

### Stardew data

- 505 versioned structured records with provenance;
- 30 Standard Bundles and explicit partial handling for Remixed requests;
- 317 acquisition relations;
- 25 offline guide pages and 100 retrievable chunks;
- 100 bilingual regression cases with controlled categories and statuses;
- 176 evidence-conditioned training examples with formal evaluation isolation;
- an auditable cleanup of the teammate's 1,262 AI-assisted candidates.

### Demonstration assets

- `demo/stardew_showcase.html`;
- `results/stardew/demo_outputs.md`, generated locally by the release builder;
- `notebooks/04_stardew_release_demo.ipynb`;
- JSON build/evaluation/validation evidence for every displayed number.

The final report and slide deck are intentionally deferred and are not part of this code patch.

## Model tracks

### Fixed target

```text
Qwen3-4B
```

The 4B target is responsible for final natural-language quality. The repository includes grounded prompting, validation, QLoRA configuration, and evaluation tooling.

### Reliable draft baseline

```text
Qwen3-0.6B -> Qwen3-4B
```

This is the primary speculative-decoding baseline because it already has language-modeling competence and a compatible tokenizer contract.

### Custom research draft

```text
TinyQwenDraft -> Qwen3-4B
```

The custom decoder uses target token IDs, tied embeddings, RMSNorm, GQA, RoPE, SwiGLU, and persistent KV cache. It is trained on target-generated grounded continuations. Its value must be judged by end-to-end speed and acceptance, not by training loss alone.

## Claim boundary

The following claims are supported by reproducible offline artifacts:

- data counts and distributions;
- SQLite integrity;
- deterministic routing/retrieval/rendering behavior;
- 100-case Stardew regression result;
- 184-test repository result;
- tokenizer-contract, cache, trainer, and speculative-decoder software tests.

The following are intentionally not claimed without GPU execution:

- QLoRA quality improvement;
- trained TinyQwenDraft quality;
- draft acceptance rate;
- TTFT/TPOT or throughput gains;
- end-to-end speculative-decoding speedup.

The formal Stardew regression records are `machine_validated` and require an independent human source review before they can be described as human-approved.
