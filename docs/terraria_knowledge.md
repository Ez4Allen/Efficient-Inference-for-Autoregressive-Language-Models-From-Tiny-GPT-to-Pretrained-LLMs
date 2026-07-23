# Terraria structured knowledge architecture

## Pipeline

```text
Cargo API -> raw -> normalized -> cleaned -> linked -> integrity audit -> SQLite
```

The repository tracks the cleaned snapshot. The linked layer and SQLite query
database are deterministic derived artifacts.

## Public interfaces

### `TerrariaQueryStore`

Read-only, indexed access to Items, NPCs, Recipes, reverse recipe relations,
and Drops. Same-name Items can be disambiguated with `item_id` or
`internal_name`; same-name NPC families can be disambiguated with `npc_id`.

### `TerrariaFactService`

Transforms query rows into compact domain facts with:

- normalized probability and quantity displays;
- explicit ambiguity/family results;
- warnings for unresolved legacy references;
- provenance containing catalog IDs.

## Snapshot validation

`snapshot_manifest.json` keeps dataset-specific expectations outside the code.
The build pipeline validates hashes and counts in strict mode. The integrity
auditor itself remains data-driven and can validate a refreshed catalog without
hard-coded record totals.

## Known partial references

Some legacy or platform-specific entities are intentionally unresolved rather
than incorrectly linked to modern IDs. These are reported as partial records;
they do not create dangling SQLite foreign keys.
