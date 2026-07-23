# Grounded Terraria Assistant

## Purpose

The assistant converts ordinary English or Chinese Terraria questions into
structured catalog queries, preserves ambiguity instead of guessing, and
renders an answer from `TerrariaFactService` evidence.

```text
user question
    -> IntentRouter
    -> EntityResolver
    -> StructuredRetriever
    -> TerrariaFactService
    -> ContextBuilder
    -> deterministic renderer or injected grounded generator
    -> answer + warnings + evidence
```

The structured catalog remains the source of factual data. The assistant does
not add recipes, drops, NPC statistics, or item properties of its own.

## Supported intents

- `item`
- `npc`
- `recipe`
- `recipes_using_item`
- `drops_for_item`
- `drops_from_source`
- `search`

Examples:

```text
How do I craft Night's Edge?
What does Armored Skeleton drop in expert mode?
Where can I get Beam Sword?
What can I craft with Terra Blade?
Moon Lord 属性是什么？
夜之刃怎么合成？
```

A small alias layer maps common Chinese surface names such as `夜之刃` and
`月亮领主` to canonical catalog names. The aliases only resolve names; all
returned facts still come from the SQLite catalog.

## Python API

```python
from src.assistant import TerrariaAssistant

with TerrariaAssistant(auto_build=True) as assistant:
    response = assistant.answer("夜之刃怎么合成？")
    print(response.answer)
    print(response.evidence)
```

Use `response.context.text` as a grounded prompt for an external language
model. A generator can also be injected without changing routing or retrieval:

```python
from src.assistant import CallableAnswerGenerator, TerrariaAssistant


def generate(context, fallback):
    # Call a local or remote model using context.text.
    return fallback


with TerrariaAssistant(
    auto_build=True,
    generator=CallableAnswerGenerator(generate),
) as assistant:
    print(assistant.answer("What does Moon Lord drop?").answer)
```

## CLI

```bash
python scripts/build_terraria_knowledge.py --quiet
python scripts/chat_terraria.py "How do I craft Night's Edge?"
python scripts/chat_terraria.py "装甲骷髅掉什么？" --mode expert
python scripts/chat_terraria.py "What is Terra Blade?" --json
python scripts/chat_terraria.py --context "Where can I get Beam Sword?"
```

Omit the question to start an interactive terminal session.

## Grounding behavior

- Same-name Items return a clarification request unless `item_id` or
  `internal_name` is supplied.
- Same-name NPC families return a clarification request unless `npc_id` is
  supplied.
- Unknown entities produce a grounded not-found response and optional catalog
  suggestions; they are not fabricated.
- Recipe questions use preferred/current variants by default. Pass
  `preferred_only=False` or `--all-variants` to include legacy variants.
- Every answer retains catalog provenance in `response.evidence`.

## Current boundary

This first assistant release answers structured Item, NPC, Recipe, and Drop
questions. Long-form progression and mechanics questions such as "What should I
do on the first night?" require a separate document-retrieval corpus and are
intentionally not answered from unsupported structured evidence.
