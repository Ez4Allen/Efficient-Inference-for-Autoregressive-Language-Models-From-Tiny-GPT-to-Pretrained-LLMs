# Stardew evidence-conditioned training data

The v1 training set is generated deterministically from the structured catalog:

```text
stardew_grounded_train_v1.jsonl       159
stardew_grounded_validation_v1.jsonl   17
```

Each prompt includes explicit evidence and requires a source citation. Only assistant continuation tokens should contribute to the language-model loss. Formal evaluation files are excluded and the validator checks for question overlap.

Regenerate with:

```bash
python scripts/generate_stardew_release_data.py
```
