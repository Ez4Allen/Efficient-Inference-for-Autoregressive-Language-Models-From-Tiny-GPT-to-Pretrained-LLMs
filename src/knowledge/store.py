
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.knowledge.normalizer import normalize_entity_name
from src.knowledge.validator import validate_record


class StructuredKnowledgeStore:
    """
    Store and query structured Terraria facts.

    The store uses exact normalized names and aliases.
    It does not perform fuzzy matching.
    """

    def __init__(
        self,
        records: list[dict[str, Any]],
    ) -> None:
        # id -> complete record
        self.records_by_id: dict[str, dict[str, Any]] = {}

        # normalized entity name -> one or more record ids
        self.alias_index: dict[str, set[str]] = defaultdict(set)

        for record in records:
            self._add_record(record)

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
    ) -> "StructuredKnowledgeStore":
        """
        Load records from one JSONL file.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Knowledge file not found: {path}"
            )

        records: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                # Ignore empty lines.
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at "
                        f"{path}:{line_number}"
                    ) from error

                records.append(record)

        return cls(records)


    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
    ) -> "StructuredKnowledgeStore":
        """
        Load and combine all JSONL knowledge files
        inside one directory.

        Files are loaded in sorted filename order.
        Duplicate record ids are rejected when the
        combined store is created.
        """
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Knowledge directory not found: "
                f"{directory}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Knowledge path is not a directory: "
                f"{directory}"
            )

        jsonl_files = sorted(
            directory.glob("*.jsonl")
        )

        if not jsonl_files:
            raise FileNotFoundError(
                f"No JSONL files found in: "
                f"{directory}"
            )

        records: list[dict[str, Any]] = []

        for path in jsonl_files:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                for line_number, line in enumerate(
                    file,
                    start=1,
                ):
                    line = line.strip()

                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(
                            f"Invalid JSON at "
                            f"{path}:{line_number}"
                        ) from error

                    records.append(record)

        return cls(records)

    def _add_record(
        self,
        record: dict[str, Any],
    ) -> None:
        """
        Validate one record and add it to the indexes.
        """
        # Validate the complete record before adding it to the indexes.
        validate_record(record)

        required_fields = {
            "id",
            "fact_type",
            "entity_name",
        }

        missing_fields = (
            required_fields - record.keys()
        )

        if missing_fields:
            raise ValueError(
                "Record is missing required fields: "
                f"{sorted(missing_fields)}"
            )

        record_id = record["id"]

        if record_id in self.records_by_id:
            raise ValueError(
                f"Duplicate record id: {record_id}"
            )

        # Save the complete record by its unique id.
        self.records_by_id[record_id] = record

        # Canonical name and aliases all point to this record.
        names = [
            record["entity_name"],
            *record.get("aliases", []),
        ]

        for name in names:
            normalized_name = normalize_entity_name(
                name
            )

            if normalized_name:
                self.alias_index[
                    normalized_name
                ].add(record_id)

    def lookup(
        self,
        entity_name: str,
        fact_type: str | None = None,
        require_verified: bool = False,
    ) -> dict[str, Any]:
        """
        Look up a record by canonical name or alias.

        Possible statuses:
        - found
        - not_found
        - ambiguous
        """
        normalized_query = normalize_entity_name(
            entity_name
        )

        record_ids = self.alias_index.get(
            normalized_query,
            set(),
        )

        candidates = [
            self.records_by_id[record_id]
            for record_id in sorted(record_ids)
        ]

        if fact_type is not None:
            candidates = [
                record
                for record in candidates
                if record["fact_type"] == fact_type
            ]

        if require_verified:
            candidates = [
                record
                for record in candidates
                if record.get("verified") is True
            ]

        if len(candidates) == 0:
            return {
                "status": "not_found",
                "query": entity_name,
                "fact_type": fact_type,
                "record": None,
                "candidates": [],
            }

        if len(candidates) > 1:
            return {
                "status": "ambiguous",
                "query": entity_name,
                "fact_type": fact_type,
                "record": None,
                "candidates": candidates,
            }

        return {
            "status": "found",
            "query": entity_name,
            "fact_type": fact_type,
            "record": candidates[0],
            "candidates": [],
        }

    def get_by_id(
        self,
        record_id: str,
    ) -> dict[str, Any] | None:
        """
        Return one record directly by its unique id.
        """
        return self.records_by_id.get(record_id)

    def list_fact_types(self) -> list[str]:
        """
        Return all available fact types.
        """
        fact_types = {
            record["fact_type"]
            for record in self.records_by_id.values()
        }

        return sorted(fact_types)

    def __len__(self) -> int:
        """
        Return the total number of records.
        """
        return len(self.records_by_id)
