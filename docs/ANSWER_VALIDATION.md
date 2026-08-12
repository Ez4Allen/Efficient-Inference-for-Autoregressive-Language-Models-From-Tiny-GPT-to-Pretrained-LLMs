# Answer Validation and Evaluation Scoring

This document answers the course feedback asking how GameGuideLM decides that an
answer is valid.  The project deliberately separates **runtime grounding
validation** from **offline benchmark scoring**.  They serve different purposes
and both are reported.

## 1. Runtime grounding validator

The online pipeline first retrieves a bounded evidence package and gives every
selected source a stable ID such as `[S1]`.  After Qwen generates an answer,
`src/gameguide/validation.py` checks:

1. every citation refers to a source ID that was actually shown in the prompt;
2. a factual paragraph contains a citation when evidence is available;
3. URLs are restricted to retrieved sources;
4. high-risk numeric claims occur in the evidence support payload;
5. the answer is non-empty and below `max_answer_chars`;
6. hidden reasoning markers such as `<think>` are absent.

One constrained repair attempt may use the same evidence.  The safe deployment
path can return a deterministic renderer answer if repair also fails.  Raw-model
quality experiments disable fallback so a failed generation remains visible.

## 2. Offline benchmark annotation

Each evaluation record includes:

```text
expected_status
intent
required_facts / must_include
forbidden_errors / must_not_include
reference_answer
```

`required_facts` can be strings or logical objects:

```json
{"any_of": ["Summer", "夏季"]}
{"all_of": ["rain", "river"]}
```

## 3. Required-fact matching

Text is normalized with Unicode NFKC, case folding, punctuation removal, and
common time-format normalization.  Matching then uses:

1. normalized substring match; otherwise
2. semantic-token recall.

Short facts containing four or fewer semantic tokens require full token recall.
Longer facts use a 0.75 token-recall threshold.  Required-fact coverage is:

\[
\text{coverage}
=\frac{\text{matched required facts}}{\text{all required facts}}.
\]

Forbidden-error rate is defined analogously over prohibited claims.

## 4. Pass criterion

The current benchmark pass rule is the conjunction:

```text
status_match
AND intent_match
AND required_fact_coverage >= 0.75
AND forbidden_error_rate == 0
AND citation_valid
AND unsupported_numeric_claims == 0
```

This project-specific score is supplemented with ROUGE-L, chrF, token F1, and
optional multilingual BERTScore.  Reference-similarity metrics are not used as a
replacement for fact checking because a fluent paraphrase can still contain the
wrong season, quantity, time, or recipe.

## 5. Worked example

Question:

```text
When and where can Catfish be caught?
```

Required facts:

```text
Spring or Fall
rain
river
6 AM to midnight
```

Candidate answer:

```text
Catfish can be caught in the river during Spring or Fall when it is raining,
from 6 AM to midnight. [S1]
```

The answer matches all four facts, contains no forbidden claim, cites a selected
source, and introduces no unsupported number.  Coverage is `4/4 = 1.0`, so the
record passes when status and intent also match.

Run a trace for any evaluated example:

```bash
python scripts/explain_answer_validation.py \
  --input results/evaluation/rows.jsonl \
  --id stardew_eval_001 \
  --output results/evaluation/stardew_eval_001_trace.json
```
