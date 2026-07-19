
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.knowledge.normalizer import normalize_entity_name
from src.knowledge.store import StructuredKnowledgeStore
from src.knowledge.validator import validate_record


FACT_TYPE_TO_FILENAME = {
    "recipe": "recipes.jsonl",
    "boss_summon": "boss_summons.jsonl",
}


def append_record(
    record: dict[str, Any],
    knowledge_directory: str | Path,
) -> Path:
    """
    Validate and append one structured fact.

    The function:
    1. Validates the record schema.
    2. Rejects duplicate record IDs.
    3. Rejects same-type entity-name or alias collisions.
    4. Selects the correct JSONL file from fact_type.
    5. Appends one JSON line.

    Returns:
        The JSONL path that received the record.
    """
    validate_record(record)

    knowledge_directory = Path(
        knowledge_directory
    )

    knowledge_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    fact_type = record["fact_type"]

    filename = FACT_TYPE_TO_FILENAME.get(
        fact_type
    )

    if filename is None:
        raise ValueError(
            f"No output file is configured for "
            f"fact_type {fact_type!r}."
        )

    existing_store = _load_existing_store(
        knowledge_directory
    )

    if existing_store is not None:
        _check_duplicate_id(
            record=record,
            store=existing_store,
        )

        _check_alias_collisions(
            record=record,
            store=existing_store,
        )

    output_path = (
        knowledge_directory / filename
    )

    with output_path.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )

    return output_path


def _load_existing_store(
    knowledge_directory: Path,
) -> StructuredKnowledgeStore | None:
    """
    Load the existing knowledge directory.

    Returns None when the directory has no JSONL files yet.
    """
    jsonl_files = list(
        knowledge_directory.glob("*.jsonl")
    )

    if not jsonl_files:
        return None

    return StructuredKnowledgeStore.from_directory(
        knowledge_directory
    )


def _check_duplicate_id(
    record: dict[str, Any],
    store: StructuredKnowledgeStore,
) -> None:
    """
    Reject an ID that already exists anywhere
    in the structured knowledge directory.
    """
    record_id = record["id"]

    if store.get_by_id(record_id) is not None:
        raise ValueError(
            f"Duplicate record id: {record_id}"
        )


def _check_alias_collisions(
    record: dict[str, Any],
    store: StructuredKnowledgeStore,
) -> None:
    """
    Reject names that already belong to another record
    of the same fact type.

    The same entity name may still be used by different
    fact types, such as recipe and item_drop.
    """
    fact_type = record["fact_type"]

    names = [
        record["entity_name"],
        *record.get("aliases", []),
    ]

    normalized_names = {
        normalize_entity_name(name)
        for name in names
        if normalize_entity_name(name)
    }

    for normalized_name in normalized_names:
        existing_ids = store.alias_index.get(
            normalized_name,
            set(),
        )

        for existing_id in existing_ids:
            existing_record = (
                store.records_by_id[existing_id]
            )

            if (
                existing_record["fact_type"]
                == fact_type
            ):
                raise ValueError(
                    "Entity-name collision for "
                    f"{record['entity_name']!r}: "
                    f"normalized name "
                    f"{normalized_name!r} already "
                    f"belongs to record "
                    f"{existing_id!r} with the same "
                    f"fact_type {fact_type!r}."
                )

def append_records(
    records: list[dict[str, Any]],
    knowledge_directory: str | Path,
) -> dict[str, Path]:
    """
    Validate and append multiple structured facts.

    All records are validated before any file is modified.

    Checks:
    1. Every record follows its fact-type schema.
    2. Record IDs are unique within the batch.
    3. Names and aliases do not collide within the batch.
    4. IDs do not already exist in the knowledge directory.
    5. Names and aliases do not collide with existing records.

    Returns:
        Mapping from fact_type to the JSONL file that was updated.
    """
    if not isinstance(records, list):
        raise TypeError(
            "Records must be provided as a list."
        )

    if not records:
        raise ValueError(
            "Records list cannot be empty."
        )

    knowledge_directory = Path(
        knowledge_directory
    )

    knowledge_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Validate every record before modifying any files.
    for index, record in enumerate(records):
        try:
            validate_record(record)
        except (ValueError, TypeError) as error:
            raise type(error)(
                f"Invalid record at batch index {index}: "
                f"{error}"
            ) from error

    _check_batch_duplicate_ids(records)
    _check_batch_alias_collisions(records)

    existing_store = _load_existing_store(
        knowledge_directory
    )

    if existing_store is not None:
        for record in records:
            _check_duplicate_id(
                record=record,
                store=existing_store,
            )

            _check_alias_collisions(
                record=record,
                store=existing_store,
            )

    records_by_file: dict[str, list[dict[str, Any]]] = {}

    for record in records:
        fact_type = record["fact_type"]

        filename = FACT_TYPE_TO_FILENAME.get(
            fact_type
        )

        if filename is None:
            raise ValueError(
                f"No output file is configured for "
                f"fact_type {fact_type!r}."
            )

        records_by_file.setdefault(
            filename,
            [],
        ).append(record)

    updated_paths: dict[str, Path] = {}

    for filename, file_records in records_by_file.items():
        output_path = (
            knowledge_directory / filename
        )

        with output_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            for record in file_records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        fact_type = file_records[0]["fact_type"]
        updated_paths[fact_type] = output_path

    return updated_paths


def _check_batch_duplicate_ids(
    records: list[dict[str, Any]],
) -> None:
    """
    Reject duplicate record IDs inside one batch.
    """
    seen_ids: set[str] = set()

    for record in records:
        record_id = record["id"]

        if record_id in seen_ids:
            raise ValueError(
                "Duplicate record id inside batch: "
                f"{record_id}"
            )

        seen_ids.add(record_id)


def _check_batch_alias_collisions(
    records: list[dict[str, Any]],
) -> None:
    """
    Reject same-fact-type name collisions inside one batch.
    """
    name_owners: dict[
        tuple[str, str],
        str,
    ] = {}

    for record in records:
        fact_type = record["fact_type"]
        record_id = record["id"]

        names = [
            record["entity_name"],
            *record.get("aliases", []),
        ]

        normalized_names = {
            normalize_entity_name(name)
            for name in names
            if normalize_entity_name(name)
        }

        for normalized_name in normalized_names:
            key = (
                fact_type,
                normalized_name,
            )

            existing_owner = name_owners.get(key)

            if (
                existing_owner is not None
                and existing_owner != record_id
            ):
                raise ValueError(
                    "Entity-name collision inside batch: "
                    f"normalized name "
                    f"{normalized_name!r} is used by "
                    f"both {existing_owner!r} and "
                    f"{record_id!r} for fact_type "
                    f"{fact_type!r}."
                )

            name_owners[key] = record_id

