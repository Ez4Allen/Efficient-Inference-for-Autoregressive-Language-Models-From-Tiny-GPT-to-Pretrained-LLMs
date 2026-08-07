# GameGuideLM v1.1.0

GameGuideLM v1.1.0 is the code-and-data milestone for the course project. It packages the
multi-game grounded language-model system, a complete Stardew Valley
course-release workload, a custom Qwen-token-compatible draft model, an
interactive demonstration, and reproducible validation. The final written
report and slide deck are intentionally deferred to a later milestone.

## Main contribution

The system separates game knowledge from model parameters. Terraria and
Stardew Valley expose different schemas through one model-facing evidence
contract. Structured facts, guide passages, player state, explicit safety
statuses, and provenance are assembled into grounded prompts for a Qwen3-4B
target. Qwen3-0.6B and a custom `TinyQwenDraft` are supported as speculative
drafts.

## Stardew Valley course-release workload

- 505 versioned structured records;
- 41 crops, 55 fish, 34 villagers, 117 recipes, 30 Standard Bundles, and
  228 acquisition entities;
- 317 structured acquisition relations;
- 25 project-authored offline guide summaries producing 100 searchable chunks;
- 100 deterministic regression cases with a 50/50 English-Chinese split;
- controlled `found`, `needs_context`, `partial`, and `not_found` behavior;
- 159 deterministic grounded training records and 17 validation records;
- 1,262 legacy AI-assisted SFT candidates cleaned, leakage-audited, and reset to
  `verified=false` / `review_status=pending`;
- one-command build, validation, evaluation, demo generation, and test run;
- self-contained interactive HTML showcase.

## Model and inference work

- Qwen3-4B target and Qwen3-0.6B pretrained draft runtime;
- custom `TinyQwenDraft` built directly in PyTorch;
- exact target-tokenizer fingerprint contract;
- tied token embedding and output projection;
- RMSNorm, RoPE, grouped-query attention, Q/K RMSNorm, SwiGLU, and persistent
  crop-able KV cache;
- assistant-only sequence-level target adaptation pipeline;
- persistent-cache greedy speculative decoder with target-consistent `exact`
  verification and performance-oriented `block` verification;
- target/draft agreement, top-k overlap, entropy, JS divergence, acceptance,
  and latency instrumentation.

## Reliability and reproducibility

- one command: `python scripts/build_stardew_release.py`;
- 184 offline tests passed;
- 100/100 deterministic Stardew regression cases passed;
- release manifests include counts and SHA-256 hashes;
- formal evaluation is excluded from training data;
- unsupported entities and false premises are refused rather than guessed;
- Standard and Remixed Bundle modes are not silently mixed;
- every release artifact states the remaining human-review and GPU-experiment
  boundaries.

## Demonstration assets

This code release contains:

- `demo/stardew_showcase.html`, a self-contained interactive showcase;
- `notebooks/04_stardew_release_demo.ipynb`, a Colab-friendly release demo;
- one-command build, validation, evaluation, and demo-generation scripts;
- reproducible source code, curated data snapshots, tests, and manifests.

The final report and PPTX/PDF presentation are not included in this patch and
will be produced after the remaining GPU experiments and human review.

## Deliberately not claimed

- The Stardew snapshot is a defined course-release scope, not a complete import
  of every Wiki article.
- The 100 regression records are machine-validated candidates, not an
  independently approved benchmark.
- The guide seed contains project-authored summaries with source attribution,
  not verbatim Wiki snapshots.
- No Qwen model weights, trained LoRA adapters, or trained TinyQwenDraft
  checkpoint are distributed.
- No speculative-decoding speedup is claimed without a real GPU training and
  warm-benchmark run.
- Sequence-level target adaptation is not full-logit distillation.

## Reproduce

```bash
pip install -r requirements-dev.txt
python scripts/build_stardew_release.py
```

The build regenerates the Stardew release data, cleans the legacy SFT set,
builds both local databases, runs the regression suite, validates manifests,
regenerates the demo, and runs all tests.
