# Stardew Valley Knowledge Module — Development Task Specification

## 0. Short Project Introduction

Our current project is not intended to train a model to memorize all game knowledge. Instead, each game is implemented as a searchable, verifiable knowledge workload for an LLM inference system.

The existing Terraria module already contains two evidence channels:

```text
Structured fact database
→ Items / NPCs / Recipes / Drops / other deterministic facts

Guide document database
→ Wiki cleaning / section-aware chunking / FTS retrieval / source tracking

Both evidence channels
→ Grounded prompt
→ Qwen3-4B answer generation
→ Qwen3-0.6B + Qwen3-4B speculative decoding experiments
```

Your task is to implement Stardew Valley as the second game knowledge module.

You do **not** need to modify or train the language models. Your work should focus on:

1. A Stardew Valley structured fact database;
2. A Stardew Valley guide-document retrieval database;
3. English and Chinese query support;
4. Explicit version, platform, condition, and provenance metadata;
5. A manually reviewed evaluation dataset;
6. Automated tests;
7. A standardized evidence output that can be consumed by the existing Qwen and speculative-decoding runtime.

The objective is to prove that the system is not hard-coded for Terraria.

---

# 1. Project Goal

## 1.1 Primary objective

Build an independently testable Stardew Valley knowledge module containing:

1. A structured fact database;
2. A Wiki guide-document database;
3. A query interface;
4. A fact-service interface;
5. Player-state-aware conditions;
6. A manually annotated benchmark;
7. Automated integrity and regression tests;
8. A common evidence contract compatible with the main project.

## 1.2 Out of scope

This task does **not** include:

- Training Qwen;
- QLoRA;
- Modifying `src/models/`;
- Modifying `src/inference/`;
- Modifying speculative decoding;
- Building a frontend;
- Crawling the entire Wiki;
- Downloading or committing Wiki images, game sprites, audio, or other game assets;
- Making the model memorize Stardew Valley facts;
- Creating a separate model runtime.

The team member assigned to Stardew Valley owns only the knowledge and retrieval layers.

---

# 2. Relationship to the Existing Project

The following shared modules already exist and should remain unchanged:

```text
src/models/
src/inference/
src/training/
src/assistant/qwen_generator.py
src/assistant/answer_validator.py
```

The Stardew Valley implementation should be added under separate directories:

```text
src/games/stardew/
data/stardew/
tests/games/stardew/
docs/
scripts/
```

Recommended architecture:

```text
Official Stardew Valley Wiki / manually verified facts
                    ↓
              importer / parser
                    ↓
          normalized JSONL snapshot
                    ↓
        integrity audit + SQLite build
                    ↓
      StardewQueryStore / FactService
                    ↓
            standardized evidence
                    ↓
           existing Qwen runtime
```

Guide-document path:

```text
Official Stardew Valley Wiki pages
                    ↓
             MediaWiki importer
                    ↓
               HTML cleaner
                    ↓
        section-aware text chunker
                    ↓
                SQLite FTS
                    ↓
           StardewGuideStore
                    ↓
            standardized evidence
```

---

# 3. Git Workflow

## 3.1 Branch

Start from the latest working project branch:

```bash
git checkout <latest-working-branch>
git pull
git checkout -b feature/stardew-knowledge-module
```

Recommended commit sequence:

```text
1. Add Stardew data schemas and source manifest
2. Add Stardew structured catalog pipeline
3. Add Stardew query store and fact service
4. Add Stardew guide corpus pipeline
5. Add Stardew evaluation annotations and tests
6. Add Stardew documentation and CLI
```

Do not place the entire implementation into one very large commit.

## 3.2 Files and directories that should not be modified

Unless discussed with the team first, do not modify:

```text
src/models/
src/inference/
src/training/
src/assistant/terraria_assistant.py
src/assistant/qwen_generator.py
configs/terraria_qwen3_pair.yaml
```

Do not rename existing Terraria interfaces.

---

# 4. Required Deliverable Structure

Minimum expected structure:

```text
data/stardew/
├── catalog/
│   ├── config/
│   │   └── sources.yaml
│   ├── cleaned/
│   │   ├── entities.jsonl
│   │   ├── crops.jsonl
│   │   ├── fish.jsonl
│   │   ├── villagers.jsonl
│   │   ├── gifts.jsonl
│   │   ├── recipes.jsonl
│   │   ├── bundles.jsonl
│   │   └── acquisition_sources.jsonl
│   ├── snapshot_manifest.json
│   └── ATTRIBUTION.md
│
├── guides/
│   ├── config/
│   │   └── sources.yaml
│   ├── ATTRIBUTION.md
│   └── README.md
│
└── evaluation/
    ├── stardew_validation_v1.jsonl
    └── stardew_eval_v1.jsonl

src/games/stardew/
├── __init__.py
├── schemas.py
├── normalizers.py
├── catalog_parser.py
├── database_builder.py
├── integrity.py
├── pipeline.py
├── query_store.py
├── fact_service.py
├── intent_router.py
├── aliases.py
├── guide_pipeline.py
├── guide_store.py
└── guide_query_expansion.py

scripts/
├── build_stardew_knowledge.py
├── build_stardew_guides.py
├── query_stardew.py
├── chat_stardew.py
└── validate_stardew_data.py

tests/games/stardew/
├── conftest.py
├── test_catalog_pipeline.py
├── test_query_store.py
├── test_fact_service.py
├── test_intent_router.py
├── test_guide_pipeline.py
└── test_annotation_schema.py

docs/
├── stardew_knowledge.md
└── stardew_annotation_guidelines.md
```

---

# 5. Data Sources and Compliance

## 5.1 Allowed source

The first version should use only:

```text
Official Stardew Valley Wiki
```

Allowed content:

- Wiki article text;
- Facts contained in Wiki tables;
- Page titles;
- Section titles;
- Revision IDs;
- Source URLs;
- Retrieval timestamps;
- License and attribution information.

Do not download or commit:

- Images;
- Game sprites;
- Audio;
- Full raw HTML pages;
- Proprietary assets from the game installation;
- Full text copied from unofficial third-party guides;
- Any content with unclear licensing.

## 5.2 MediaWiki access

Use the MediaWiki API when possible. Do not aggressively scrape rendered HTML pages.

Suggested configuration:

```yaml
wiki:
  name: Official Stardew Valley Wiki
  api_url: https://stardewvalleywiki.com/mediawiki/api.php
  article_base_url: https://stardewvalleywiki.com/
  language: en
  license:
    name: CC BY-NC-SA 3.0
    attribution_url: https://stardewvalleywiki.com/Stardew_Valley_Wiki:Copyrights
  request:
    user_agent: EfficientInferenceStardewResearch/0.1 (+PROJECT_GITHUB_URL)
    timeout_seconds: 30
    request_delay_seconds: 0.75
    max_retries: 4
    checkpoint_every: 5
```

Before a full import, run a three-page smoke test.

The build report must record:

```text
API endpoint
source pages requested
source pages retrieved
missing pages
request failures
revision IDs
license name
retrieval timestamp
```

## 5.3 Version metadata

Every structured record must include:

```text
game_version
platform
version_notes
```

A target version should be declared in the snapshot manifest.

However, a global target version is not sufficient. Platform-specific or legacy differences must also be represented at record level.

Supported platform values:

```text
all
pc
console
mobile
legacy
unknown
```

Use `unknown` when the source does not support a reliable conclusion. Do not guess.

---

# 6. Structured Fact Database

## 6.1 Why the Terraria schema cannot simply be copied

Stardew Valley facts depend heavily on:

- Season;
- Calendar day;
- Time of day;
- Weather;
- Location;
- Year;
- Community Center or Joja route;
- Standard or Remixed Bundles;
- Skill level;
- Friendship level;
- Farm type;
- Platform and game version.

These conditions are different from Terraria difficulty and progression conditions.

The Stardew Valley data should therefore use a game-specific schema while preserving a common output contract.

## 6.2 Common fields for every cleaned record

Every cleaned JSONL record must contain at least:

```json
{
  "schema_version": 1,
  "game": "stardew_valley",
  "game_version": "TARGET_VERSION",
  "platform": "all",
  "record_type": "crop",
  "source_catalog_id": "stardew:crop:canonical-name",
  "name": "Canonical English Name",
  "normalized_name": "canonicalenglishname",
  "aliases": ["optional alias"],
  "facts": {},
  "conditions": {},
  "provenance": {
    "source_name": "Official Stardew Valley Wiki",
    "page_title": "PAGE_TITLE",
    "section_title": "SECTION_TITLE",
    "source_url": "SOURCE_URL",
    "revision_id": null,
    "retrieved_at": "ISO-8601",
    "license_name": "CC BY-NC-SA 3.0"
  },
  "parse_status": "ok",
  "parse_warnings": []
}
```

Rules:

- `source_catalog_id` must be stable and unique;
- `normalized_name` is only for lookup and must not replace the canonical name;
- Missing values must be represented as `null`;
- Do not use `0`, an empty string, or `"unknown"` to hide missing data;
- Any manual correction must appear in `parse_warnings` or a dedicated `manual_override` field;
- If provenance or parsing is incomplete, use `parse_status: "partial"`;
- Do not add facts based on personal knowledge.

## 6.3 Standard condition structure

Use the following condition fields when applicable:

```json
{
  "seasons": [],
  "weather": [],
  "time_start": null,
  "time_end": null,
  "locations": [],
  "day_min": null,
  "day_max": null,
  "year_min": null,
  "year_max": null,
  "skill": null,
  "skill_level_min": null,
  "friendship_hearts_min": null,
  "route": "any",
  "bundle_mode": "any",
  "farm_types": [],
  "notes": []
}
```

Critical conditions must be represented structurally. They must not exist only as free text.

---

# 7. MVP Structured Data Scope

## 7.1 Crops

File:

```text
crops.jsonl
```

Minimum fields:

```text
crop_name
seed_name
seasons
growth_days
regrow_days
harvest_quantity
trellis
giant_crop
seed_buy_price
base_sell_price
locations_or_exceptions
```

Minimum coverage:

```text
At least 30 crop records
```

The system should be able to answer:

```text
Which season can this crop grow in?
How many days does it take to mature?
Does it regrow?
Does it require a trellis?
Can it still be harvested if planted on a given day?
```

Crop-deadline questions must be calculated from season day and growth time. They must not be guessed by the LLM.

## 7.2 Fish

File:

```text
fish.jsonl
```

Minimum fields:

```text
fish_name
seasons
weather
time_windows
locations
fishing_zone_or_special_location
difficulty
behavior
base_sell_prices
special_conditions
```

Minimum coverage:

```text
At least 50 fish records
```

If a fish has multiple combinations of season, weather, time, and location, represent them as separate `availability_windows`.

Example:

```json
{
  "availability_windows": [
    {
      "seasons": ["spring", "fall"],
      "weather": ["rain"],
      "time_start": "06:00",
      "time_end": "24:00",
      "locations": ["river"]
    }
  ]
}
```

Do not combine several distinct availability rules into one unparseable sentence.

## 7.3 Villagers and gifts

Files:

```text
villagers.jsonl
gifts.jsonl
```

Villager fields:

```text
villager_name
birthday_season
birthday_day
marriageable
home
relations_or_notes
```

Gift-relation fields:

```text
villager_name
item_name
preference
conditions
exceptions
```

Allowed preference values:

```text
love
like
neutral
dislike
hate
```

Minimum coverage:

```text
At least 20 villagers
Every included villager must have queryable loved gifts
```

Universal Loves and Universal Likes must be represented separately from individual exceptions.

Do not overwrite personal exceptions with universal rules.

## 7.4 Recipes

File:

```text
recipes.jsonl
```

Recipe types:

```text
cooking
crafting
```

Required fields:

```text
recipe_type
result_name
result_quantity
ingredients
unlock_source
skill_requirement
friendship_requirement
purchase_source
purchase_price
version_notes
```

Minimum coverage:

```text
At least 100 recipes
```

Each ingredient must be a structured object:

```json
{
  "item_name": "ITEM",
  "quantity": 1,
  "category_substitution": null
}
```

Category ingredients such as `Any Fish` must not be linked to one specific fish.

## 7.5 Bundles

File:

```text
bundles.jsonl
```

Bundle modes must be distinguished:

```text
standard
remixed
missing_bundle
```

Required fields:

```text
bundle_mode
room
bundle_name
requirements
selection_rule
reward
room_reward
route
```

Requirement structure:

```json
{
  "item_name": "ITEM",
  "quantity": 1,
  "minimum_quality": null,
  "optional_group": null
}
```

Minimum acceptance requirement:

```text
Complete coverage of Standard Bundles
Remixed Bundles must be distinguishable from Standard Bundles
```

If Remixed Bundle coverage is incomplete in the first version, the service must return `partial` or an explicit warning.

It must never return Standard Bundle data as if it were Remixed Bundle data.

## 7.6 Acquisition sources

File:

```text
acquisition_sources.jsonl
```

This relation table should answer where and how an entity can be obtained.

Fields:

```text
entity_name
source_type
source_name
location
season
weather
time
price
probability
quantity
conditions
```

Suggested `source_type` values:

```text
shop
foraging
crop
fishing
monster
crafting
cooking
reward
gift
machine
animal
quest
festival
```

Minimum coverage:

```text
At least 150 acquisition relations
```

---

# 8. SQLite Database

Recommended database path:

```text
data/stardew/catalog/stardew_query.sqlite3
```

This is a generated file and must not be committed.

Minimum tables:

```text
metadata
entities
aliases
crops
fish
fish_availability
villagers
gifts
recipes
recipe_ingredients
bundles
bundle_requirements
acquisition_sources
```

Add an FTS5 table:

```text
entity_fts
```

It should index at least:

```text
name
aliases
record_type
```

The original normalized JSON record may also be stored, but commonly filtered fields must have dedicated columns. Do not require all queries to parse JSON.

## 8.1 Integrity checks

The build must verify:

- `source_catalog_id` is unique;
- Aliases are not empty strings;
- Recipe ingredient quantities are positive;
- Fish time windows are valid;
- Calendar days are within 1–28;
- Season values are valid;
- Gift-preference values are valid;
- Bundle requirement quantities are positive;
- Relation records resolve to known entities;
- Unresolved relations are recorded as `partial`;
- `PRAGMA integrity_check` returns `ok`;
- The build report records input, output, skipped, warning, and error counts.

---

# 9. QueryStore and FactService

## 9.1 StardewQueryStore

Implement:

```python
class StardewQueryStore:
    def get_entity(
        self,
        name: str,
        *,
        record_type: str | None = None,
    ): ...

    def get_crop(self, name: str): ...

    def get_fish(self, name: str): ...

    def get_villager(self, name: str): ...

    def gifts_for_villager(
        self,
        name: str,
        *,
        preference: str = "love",
    ): ...

    def get_recipe(
        self,
        result_name: str,
        *,
        recipe_type: str | None = None,
    ): ...

    def recipes_using_item(
        self,
        item_name: str,
    ): ...

    def get_bundle(
        self,
        bundle_name: str,
        *,
        bundle_mode: str = "standard",
    ): ...

    def bundles_requiring_item(
        self,
        item_name: str,
        *,
        bundle_mode: str = "standard",
    ): ...

    def acquisition_sources(
        self,
        entity_name: str,
        *,
        player_state: dict | None = None,
    ): ...

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
    ): ...
```

Allowed result statuses:

```text
found
not_found
ambiguous
partial
needs_context
```

## 9.2 StardewFactService

Implement one unified entry point:

```python
class StardewFactService:
    def query(
        self,
        intent: str,
        entity: str,
        *,
        player_state: dict | None = None,
        limit: int = 10,
    ) -> dict:
        ...
```

Output contract:

```json
{
  "game": "stardew_valley",
  "status": "found",
  "intent": "fish_availability",
  "query": "USER_QUERY",
  "entity": "CANONICAL_ENTITY",
  "facts": {},
  "candidates": [],
  "warnings": [],
  "provenance": [
    {
      "entity_type": "fish",
      "source_catalog_id": "stardew:fish:...",
      "source_url": "SOURCE_URL",
      "page_title": "PAGE_TITLE",
      "section_title": "SECTION_TITLE",
      "revision_id": null,
      "game_version": "TARGET_VERSION",
      "platform": "all"
    }
  ]
}
```

The result style should remain close to the existing Terraria retrieval output so that the same ContextBuilder and Qwen generation layer can consume it.

---

# 10. Intent Router

Minimum supported intents:

```text
entity
crop_info
crop_deadline
fish_availability
villager_info
villager_gifts
recipe
recipes_using_item
bundle
bundles_requiring_item
acquisition
guide
search
unknown
```

Support English and Chinese queries.

Examples:

```text
Where and when can I catch Catfish?
What weather do I need for Catfish?
What gifts does Abigail love?
How do I craft a Quality Sprinkler?
Which Bundle requires this item?
Can I still harvest this crop if I plant it on Spring 20?
What should I prioritize during the first spring?
What should I plant today?
```

If a recommendation query lacks necessary state, return:

```text
needs_context
```

and identify the missing fields.

Do not use an LLM for the first version of the intent router.

Use deterministic rules and explicit player-state fields.

---

# 11. Player State

Use this common player-state structure:

```json
{
  "game_version": "TARGET_VERSION",
  "platform": "pc",
  "season": null,
  "day": null,
  "year": null,
  "weather": null,
  "time": null,
  "location": null,
  "route": null,
  "bundle_mode": null,
  "farm_type": null,
  "skills": {},
  "friendship_hearts": {},
  "budget": null,
  "goal": null
}
```

Rules:

- Fields may remain null if they are irrelevant to the query;
- If a field is required but missing, return `needs_context`;
- Do not silently assume first year, clear weather, Standard Bundles, or a specific route;
- The response should identify which player-state conditions were used.

---

# 12. Guide Document Database

## 12.1 Initial page scope

Do not crawl the entire Wiki.

Start with 20–40 high-value pages.

Candidate topics:

```text
Getting Started
Day Cycle
Energy
Skills
Farming
Crops
Fishing
Fish
The Mines
Combat
Foraging
Bundles
Community Center
Friendship
Villagers
Marriage
Cooking
Crafting
Quests
Special Orders
Festivals
Weather
Seasons
Farm Maps
Greenhouse
Museum
Animals
Artisan Goods
Skull Cavern
Ginger Island
```

Before adding a page to the manifest, verify the exact canonical page title using the MediaWiki API.

If the real page title differs, use the canonical title returned by the API.

## 12.2 Reusing existing retrieval infrastructure

The following generic capabilities may be reused:

```text
src/retrieval/wiki_client.py
src/retrieval/wiki_importer.py
src/retrieval/wiki_cleaner.py
src/retrieval/text_chunker.py
src/retrieval/quality_audit.py
```

Do not directly reuse Terraria-specific logic such as:

```text
src/retrieval/query_expansion.py
build_terraria_guides()
```

These currently contain Terraria-specific concepts and paths.

Add:

```text
src/games/stardew/guide_pipeline.py
src/games/stardew/guide_query_expansion.py
src/games/stardew/guide_store.py
```

## 12.3 StardewGuideStore output

Each search hit must include:

```text
chunk_id
document_id
page_title
section_title
section_path
text
source_url
revision_id
quality_status
quality_flags
score
rank
matched_terms
retrieval_role
content_kind
table_density
```

The main project will later map these records to evidence labels such as `[S1]` and `[S2]`.

## 12.4 Query expansion

At minimum, cover English and Chinese expressions for:

```text
first spring
getting started
community center
bundles
fishing
mines
skull cavern
friendship
gifts
marriage
greenhouse
ginger island
crop profit
professions
```

Do not copy Terraria stage profiles or Hardmode/Boss-related rules.

---

# 13. Data Annotation

Annotation has two parts:

```text
A. Structured-fact annotation
B. Question-answer evaluation annotation
```

## 13.1 Structured-fact annotation procedure

For each record:

1. Identify the source page;
2. Identify the exact section or table;
3. Extract the fields;
4. Normalize the canonical name;
5. Add aliases;
6. Convert conditions into structured fields;
7. Add provenance;
8. Mark uncertainty as `partial`;
9. Run integrity validation;
10. Have another person review important records.

Rules:

- Do not write the answer first and search for support afterward;
- Do not add personal gameplay knowledge;
- Do not hide important conditions inside prose;
- Do not silently correct parser output;
- Do not infer unsupported values;
- Keep the original source reference.

Manual correction format:

```json
{
  "manual_override": {
    "applied": true,
    "reason": "SOURCE_TABLE_AMBIGUITY",
    "reviewer": "NAME",
    "reviewed_at": "ISO-8601"
  }
}
```

## 13.2 QA annotation format

One JSON object per line:

```json
{
  "schema_version": 1,
  "id": "sdv_eval_fish_0001",
  "game": "stardew_valley",
  "game_version": "TARGET_VERSION",
  "platform": "all",
  "language": "en",
  "question": "Where and when can I catch Catfish?",
  "intent": "fish_availability",
  "entities": [
    {
      "text": "Catfish",
      "canonical_name": "Catfish",
      "entity_type": "fish"
    }
  ],
  "player_state": {
    "season": null,
    "day": null,
    "year": null,
    "weather": null,
    "time": null,
    "location": null,
    "route": null,
    "bundle_mode": null
  },
  "expected_status": "found",
  "required_facts": [
    {
      "field": "availability_windows",
      "value": "SOURCE_DERIVED_VALUE"
    }
  ],
  "required_sources": [
    {
      "page_title": "PAGE_TITLE",
      "section_keywords": ["KEYWORD"]
    }
  ],
  "reference_answer": "A concise answer containing only source-supported facts.",
  "must_include": ["REQUIRED_CONCEPT"],
  "must_not_include": ["KNOWN_WRONG_OR_UNSUPPORTED_CLAIM"],
  "difficulty": "medium",
  "split": "eval",
  "annotator": "NAME",
  "reviewer": null,
  "review_status": "pending"
}
```

Do not use chunk ID as the only required source identifier because chunk IDs may change after re-chunking.

Use:

```text
page_title
section_keywords
```

## 13.3 Status definitions

Allowed values:

```text
found
not_found
ambiguous
needs_context
partial
```

Definitions:

### `found`

The entity exists and the available evidence is sufficient.

### `not_found`

The requested entity or fact does not exist in the current knowledge snapshot.

### `ambiguous`

The query maps to multiple possible entities or interpretations.

### `needs_context`

The answer depends on missing player-state information.

Example:

```text
What should I plant today?
```

This requires at least season and day, and may also require budget and goal.

### `partial`

The module contains only partial support, such as incomplete Remixed Bundle coverage or platform-specific uncertainty.

## 13.4 Evaluation-set size

Minimum:

```text
100 QA records
```

Recommended intent distribution:

```text
20 crop / seasonal-planning questions
15 fish-availability questions
15 villager / gift questions
15 recipe / ingredient questions
15 bundle / Community Center questions
10 acquisition questions
10 guide / progression questions
```

Minimum status distribution:

```text
70 found
10 needs_context
10 ambiguous or partial
10 not_found or false-premise
```

Language distribution:

```text
At least 50 Chinese questions
At least 30 English questions
The remaining questions may use either language
```

## 13.5 Dataset split

No QLoRA training is required at this stage.

Use:

```text
stardew_validation_v1.jsonl: 40 records
stardew_eval_v1.jsonl: 60 records
```

Do not repeatedly tune implementation rules against the final evaluation file.

If training is added later, create a separate:

```text
stardew_train_v1.jsonl
```

Do not copy evaluation examples directly into training data.

## 13.6 Two-person review

Every formal evaluation record should contain:

```text
annotator
reviewer
review_status
```

Reviewer checklist:

- Is the question natural?
- Is the canonical entity correct?
- Are required facts supported by the source?
- Does the reference answer contain unsupported information?
- Are version, platform, season, time, weather, and route conditions correct?
- Is the expected status correct?
- Is a `not_found` example genuinely absent?
- Is a `needs_context` example genuinely underspecified?

Only records with:

```text
review_status: approved
```

may enter the final evaluation set.

---

# 14. Automated Testing

All tests must run offline.

Use small synthetic fixtures rather than the live Wiki.

## 14.1 Catalog tests

Test:

- JSONL schema validation;
- Unique IDs;
- SQLite build;
- SQLite integrity check;
- Exact lookup;
- Alias lookup;
- Ambiguous lookup;
- Not-found behavior;
- FTS search;
- Provenance preservation.

## 14.2 Conditional-query tests

Test:

- Season filtering;
- Weather filtering;
- Time-window filtering;
- Crop deadline calculation;
- Standard versus Remixed Bundle filtering;
- Community Center versus Joja route;
- Missing player-state fields returning `needs_context`.

## 14.3 Guide-retrieval tests

Test:

- Removal of references, history, and navigation sections;
- Preservation of section path;
- Chunks do not begin at arbitrary mid-sentence positions;
- Chinese query expansion;
- Guide pages rank above unrelated reference tables;
- No-result behavior;
- Source URL and revision ID preservation.

## 14.4 Annotation tests

Test:

- All JSONL lines parse;
- IDs are unique;
- Enum values are valid;
- Approved evaluation records contain source references;
- `found` records contain required facts;
- `needs_context` records identify missing state fields;
- `not_found` records do not contain invented reference answers.

---

# 15. Required CLI Commands

The final module should support:

```bash
python scripts/build_stardew_knowledge.py --quiet

python scripts/build_stardew_guides.py --max-pages 3

python scripts/build_stardew_guides.py --offline

python scripts/query_stardew.py "Catfish"

python scripts/chat_stardew.py \
  "Where, when, and under what weather can I catch Catfish?"

python scripts/chat_stardew.py \
  "What gifts does Abigail love?"

python scripts/chat_stardew.py \
  "What should I prioritize during the first spring?"

python scripts/validate_stardew_data.py

python -m pytest -q tests/games/stardew

python -m pytest -q
```

The first version of `chat_stardew.py` may use a deterministic renderer. It does not need to load Qwen.

---

# 16. `.gitignore` Requirements

The repository currently ignores many JSONL files by default, so Stardew compact snapshots and evaluation files may require explicit include rules.

Files that should be committed:

```text
data/stardew/catalog/cleaned/*.jsonl
data/stardew/catalog/snapshot_manifest.json
data/stardew/catalog/ATTRIBUTION.md
data/stardew/guides/config/sources.yaml
data/stardew/guides/ATTRIBUTION.md
data/stardew/guides/README.md
data/stardew/evaluation/*.jsonl
```

Files that should not be committed:

```text
data/stardew/catalog/*.sqlite3
data/stardew/catalog/linked/
data/stardew/catalog/reports/
data/stardew/guides/raw/
data/stardew/guides/cleaned/
data/stardew/guides/chunks/
data/stardew/guides/reports/
data/stardew/guides/*.sqlite3
data/stardew/guides/*.zip
```

Suggested rules:

```gitignore
# Track compact Stardew snapshots and evaluation data
!data/stardew/
!data/stardew/catalog/
!data/stardew/catalog/cleaned/
!data/stardew/catalog/cleaned/*.jsonl
!data/stardew/catalog/snapshot_manifest.json
!data/stardew/catalog/ATTRIBUTION.md
!data/stardew/guides/
!data/stardew/guides/config/
!data/stardew/guides/config/sources.yaml
!data/stardew/guides/ATTRIBUTION.md
!data/stardew/guides/README.md
!data/stardew/evaluation/
!data/stardew/evaluation/*.jsonl

data/stardew/catalog/*.sqlite3
data/stardew/catalog/linked/
data/stardew/catalog/reports/
data/stardew/guides/raw/
data/stardew/guides/cleaned/
data/stardew/guides/chunks/
data/stardew/guides/reports/
data/stardew/guides/*.sqlite3
data/stardew/guides/*.zip
```

Use:

```bash
git check-ignore -v <path>
```

to verify whether files are tracked or ignored as intended.

---

# 17. Definition of Done

The Stardew Valley module is complete only when all of the following are true:

- [ ] No model, training, or speculative-decoding code was modified;
- [ ] The Stardew module builds independently;
- [ ] The structured database passes integrity checks;
- [ ] Crops, fish, villagers/gifts, recipes, bundles, and acquisition sources are covered;
- [ ] Every fact contains version, condition, and provenance metadata;
- [ ] The guide corpus contains at least 20 high-value pages;
- [ ] English and Chinese guide retrieval work;
- [ ] QueryStore and FactService return the standardized evidence contract;
- [ ] Unknown entities do not produce fabricated answers;
- [ ] Missing player-state information returns `needs_context`;
- [ ] Standard and Remixed Bundles are not mixed;
- [ ] At least 100 QA records are annotated;
- [ ] Final evaluation records are reviewer-approved;
- [ ] All Stardew tests pass;
- [ ] The full repository test suite has no regressions;
- [ ] README and attribution files are complete;
- [ ] The pull request contains no raw Wiki HTML, images, SQLite databases, or model weights.

---

# 18. Recommended Implementation Order

## Milestone 1 — Skeleton and contracts

- Create directories;
- Define schemas;
- Create source manifest;
- Add 10 synthetic fixtures;
- Add schema tests.

## Milestone 2 — Structured database

- Parse source pages;
- Generate cleaned JSONL;
- Build SQLite;
- Add integrity audit;
- Implement QueryStore.

## Milestone 3 — FactService and router

- Implement unified query contract;
- Add English and Chinese aliases;
- Implement `needs_context`;
- Add deterministic CLI.

## Milestone 4 — Guide corpus

- Run three-page smoke build;
- Create 20–40 page manifest;
- Clean, chunk, and index;
- Implement Stardew query expansion;
- Add guide-retrieval tests.

## Milestone 5 — Annotation and delivery

- Annotate 100 QA records;
- Complete two-person review;
- Run all tests;
- Complete documentation;
- Open pull request.

---

# 19. Integration Contract for the Main Project

Before merge, the main project only needs to validate these two interfaces:

```python
facts = StardewFactService.query(...)

hits = StardewGuideStore.search(...)
```

Both should provide:

```text
status
intent or query
facts or hits
warnings
provenance
```

The main project will then connect Stardew evidence to the existing:

```text
ContextBuilder
QwenGroundedAnswerGenerator
Qwen3-0.6B / Qwen3-4B runtime
speculative-decoding benchmark
```

The Stardew module itself does not own or maintain any model.

---

# 20. Core Design Principle

```text
The model is responsible for understanding and organizing the answer.

The database is responsible for mutable and verifiable game facts.

The retriever is responsible for finding relevant evidence.

Version metadata and player state define when each fact is valid.
```

Do not attempt to train a model to memorize the entire Stardew Valley Wiki.

Do not treat copied Wiki prose as a structured fact database.

The core deliverables are:

```text
structured facts
explicit conditions
provenance
safe refusal behavior
evaluation annotations
reproducible tests
```
