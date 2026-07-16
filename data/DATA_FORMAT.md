# SFT Game QA Data Format

This document defines the shared dataset contract for supervised fine-tuning (SFT) and evaluation. The same format can be used for Terraria, Stardew Valley, Minecraft, or another game domain.

## 1. File layout

Recommended repository layout:

```text
data/
├── DATA_FORMAT.md
├── sft_example.jsonl
├── terraria/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── eval.jsonl
├── stardew_valley/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── eval.jsonl
└── minecraft/
    ├── train.jsonl
    ├── validation.jsonl
    └── eval.jsonl
```

Each `.jsonl` file must contain **one complete JSON object per line**. Do not spread one record across multiple lines.

## 2. Required fields

Training and validation records must contain:

| Field | Type | Requirement |
|---|---|---|
| `id` | string | Unique across the project |
| `messages` | list | Chat messages in valid role order |

Evaluation records may use the same `messages` format, or the explicit evaluation format described in Section 3.

Minimal valid SFT record:

```json
{"id":"sample_001","messages":[{"role":"system","content":"You are a reliable game guide assistant."},{"role":"user","content":"How do I upgrade this tool?"},{"role":"assistant","content":"Provide the verified answer here."}]}
```


## 3. Evaluation record format

The frozen evaluation set may use either the chat `messages` format or this explicit format:

```json
{"id":"game_eval_001","split":"eval","system_prompt":"You are a reliable game guide assistant.","question":"What happened in this scenario?","reference_answer":"Provide the verified reference answer.","required_facts":["fact 1"],"forbidden_errors":["error 1"],"source_urls":["https://example.com/source"],"dataset_version":"1.0"}
```

Required fields for the explicit evaluation format:

| Field | Type | Requirement |
|---|---|---|
| `id` | string | Unique across the project |
| `question` | string | Non-empty test question |
| `reference_answer` | string | Non-empty verified reference answer |

`system_prompt` is optional. `required_facts`, `forbidden_errors`, and `source_urls` are strongly recommended.

The explicit evaluation format is **not** consumed by `SFTJsonlDataset`; it is intended for baseline generation and scoring scripts.

## 4. Message rules

Allowed roles:

```text
system
user
assistant
```

Valid single-turn order:

```text
system (optional)
user
assistant
```

Valid multi-turn order:

```text
system (optional)
user
assistant
user
assistant
...
```

Rules:

1. A `system` message is optional.
2. A `system` message may only appear first.
3. After the optional system message, roles must alternate `user`, `assistant`, `user`, `assistant`, and so on.
4. The final message must be `assistant`.
5. Every `content` value must be a non-empty string.
6. Training answers should be concise, factual, and directly useful.
7. Do not include hidden reasoning or chain-of-thought in assistant answers.

## 5. Recommended metadata

The training code only requires `id` and `messages`, but the following fields are strongly recommended:

| Field | Type | Purpose |
|---|---|---|
| `split` | string | `train`, `validation`, or `eval` |
| `domain` | string | Game name, such as `terraria` |
| `category` | string | Topic group, such as `boss_progression` |
| `language` | string | Normally `en` |
| `required_facts` | list[string] | Facts a correct answer should contain |
| `forbidden_errors` | list[string] | Errors or hallucinations the answer must avoid |
| `source_urls` | list[string] | Reliable sources used for verification |
| `verified` | boolean | Must be `true` after human review |
| `reviewed_by` | string | Reviewer name or GitHub username |
| `generation_method` | string | For example `human`, `ai_assisted`, or `mixed` |
| `dataset_version` | string | Dataset version, such as `1.0` |

Recommended record:

```json
{"id":"game_train_001","split":"train","domain":"example_game","category":"tool_upgrade","language":"en","messages":[{"role":"system","content":"You are a reliable game guide assistant. Correct false premises and do not invent mechanics."},{"role":"user","content":"How do I upgrade the tool?"},{"role":"assistant","content":"Give a concise answer containing only verified mechanics."}],"required_facts":["verified fact 1","verified fact 2"],"forbidden_errors":["invented item","wrong NPC"],"source_urls":["https://example.com/verified-source"],"verified":true,"reviewed_by":"reviewer_name","generation_method":"ai_assisted","dataset_version":"1.0"}
```

## 6. Split policy

### Train

Used for gradient updates. The model is allowed to see both the question and answer.

### Validation

Used during training to monitor generalization and validation loss. Validation records must not be copied from train.

### Eval

Used only for final model comparison. Eval questions and answers must remain frozen and must never be used for training or prompt generation.

### Leakage rules

The following are not allowed across train, validation, and eval:

- Exact duplicate questions.
- Near-identical paraphrases that test the same memorized wording.
- The same scenario with only superficial word substitutions.
- Copying an eval reference answer into train.
- Generating train examples by asking an AI to rewrite the eval set.

Testing the same broad topic is acceptable, but the scenario and reasoning demand should differ.

Bad split:

```text
Train: Which boss activates Hardmode?
Eval: What boss causes Hardmode to start?
```

Better split:

```text
Train: Explain the world changes caused by entering Hardmode.
Eval: A player sees stronger enemies and a new biome after a boss fight. Diagnose what happened.
```

## 7. AI-assisted data workflow

AI may be used to create candidate records, but AI output is not a reliable source.

Required workflow:

```text
AI generates candidate QA
→ Human checks the answer against reliable sources
→ Human corrects inaccuracies and removes unsupported claims
→ Add source_urls
→ Set verified=true
→ Run scripts/validate_sft_data.py
→ Submit for code review
```

Do not:

- Treat another language model as the factual source.
- Mark a record verified without checking it.
- Copy long passages from a wiki.
- Include modded content unless the record clearly identifies the mod and version.
- Mix facts from different game versions without labeling the version.
- Include vague answers that cannot be evaluated.
- Include answers with unnecessary filler or invented details.

## 8. Answer-writing guidelines

A strong assistant answer should:

- Answer the question immediately.
- Include all required facts.
- Avoid unrelated information.
- Correct false premises explicitly.
- Distinguish vanilla content from modded content.
- Mention version/platform constraints when relevant.
- Stop when the useful answer is complete.

For stable speculative-decoding experiments, the team may use a consistent answer structure, but factual correctness takes priority over formatting.

## 9. ID convention

Recommended pattern:

```text
<domain>_<split>_<category>_<number>
```

Examples:

```text
terraria_train_progression_0001
stardew_validation_tools_0003
minecraft_eval_crafting_0012
```

IDs must never be reused.

## 10. Versioning

Do not silently overwrite a dataset used in an experiment.

Use versioned filenames or a manifest:

```text
train_v1.jsonl
train_v2.jsonl
```

Each training run should record:

- Dataset file path.
- Dataset version.
- Git commit hash.
- Base model.
- LoRA configuration.
- Random seed.

## 11. Validation commands

Basic validation:

```bash
python scripts/validate_sft_data.py \
  --train data/<domain>/train.jsonl \
  --validation data/<domain>/validation.jsonl \
  --eval data/<domain>/eval.jsonl
```

Strict validation additionally requires verification metadata:

```bash
python scripts/validate_sft_data.py \
  --train data/<domain>/train.jsonl \
  --validation data/<domain>/validation.jsonl \
  --eval data/<domain>/eval.jsonl \
  --strict
```

Fail the command on warnings as well as errors:

```bash
python scripts/validate_sft_data.py \
  --train data/<domain>/train.jsonl \
  --validation data/<domain>/validation.jsonl \
  --eval data/<domain>/eval.jsonl \
  --strict \
  --fail-on-warnings
```

## 12. Pull-request checklist

Before submitting data:

- [ ] Each JSONL line is valid JSON.
- [ ] IDs are unique.
- [ ] Message roles are valid and alternating.
- [ ] Final message is from the assistant.
- [ ] Answers were fact-checked.
- [ ] Sources are included.
- [ ] `verified` is `true`.
- [ ] Train, validation, and eval do not overlap.
- [ ] No long copied passages.
- [ ] Validator exits successfully.
