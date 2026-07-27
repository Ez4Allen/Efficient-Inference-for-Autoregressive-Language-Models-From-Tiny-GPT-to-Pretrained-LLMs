# Stardew Valley Module

The Stardew plug-in demonstrates game-specific conditions behind a common model interface.

## Structured capabilities

- crop seasons, growth time, regrowth, and season-end deadline calculation;
- fish availability windows by season, weather, time, and location;
- villager birthdays and loved gifts;
- crafting recipe ingredients and unlock level;
- Standard Community Center bundle membership.

The tracked compact snapshot contains 31 high-confidence facts. It is a reproducible starter, not a claim of complete Wiki coverage.

## Player-state behavior

Queries such as “Can I catch Eel right now?” require season, weather, time, and location. Queries such as “Can I still plant Cauliflower in time?” require season and calendar day. The system returns `needs_context` rather than silently assuming values.

## Extending with teammate data

The teammate-maintained catalog should emit the same common record fields:

```text
source_catalog_id
record_type
name
normalized_name
aliases
facts
conditions
provenance
parse_status
parse_warnings
```

Replacing or appending `data/stardew/catalog/cleaned/facts.jsonl` does not require changes to the Qwen runtime.
