# Efficient Inference for Autoregressive Language Models

Project goal: build a small GPT-style autoregressive language model, then study prefill/decode inference behavior and optimization on pretrained language models.

## Planned components

1. Tiny GPT-style LM from scratch
2. Prefill/decode benchmark framework
3. Workload and model scaling experiments
4. Prefill/decode separation or scheduling simulation
5. Final analysis, report, and presentation

## Repository structure

```text
configs/              Experiment configuration files
src/tiny_lm/          Tiny GPT-style language model implementation
src/benchmark/        Prefill/decode benchmark utilities
src/optimization/     Scheduling and prefill/decode simulation code
scripts/              Command-line entry points
notebooks/            Colab demos and analysis notebooks
data/                 Dataset notes and small sample data only
results/              Generated experiment outputs
report/               Proposal, notes, report drafts, and figures
docs/                 Project plan and documentation
```
