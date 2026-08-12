"""Build a provenance-aware, evaluation-decontaminated project corpus.

The corpus is deliberately local and reproducible.  It combines only tracked
GameGuideLM training resources: Stardew/Terraria training conversations,
curated guide text, and structured catalog records.  Formal validation/eval
questions are never inserted.  When entities are available in held-out records,
the builder can additionally exclude matching catalog/guide documents to create
an entity-held-out study.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from bs4 import BeautifulSoup

from src.utils.io import read_jsonl, write_json, write_jsonl


def normalize_key(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", "", text)


def infer_language(text: str) -> str:
    return "zh" if any("\u3400" <= character <= "\u9fff" for character in text) else "en"


def flatten_entity_values(value: Any) -> list[str]:
    """Normalize heterogeneous benchmark entity schemas into scalar labels."""

    if value is None:
        return []
    if isinstance(value, Mapping):
        items: list[str] = []
        for child in value.values():
            items.extend(flatten_entity_values(child))
        return items
    if isinstance(value, (list, tuple, set)):
        items = []
        for child in value:
            items.extend(flatten_entity_values(child))
        return items
    rendered = str(value).strip()
    return [rendered] if rendered else []


def stable_split(source_id: str, *, validation_fraction: float, seed: int) -> str:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    digest = hashlib.sha256(f"{seed}:{source_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return "validation" if value < validation_fraction else "train"


def _flatten_fact(value: Any, *, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten_fact(child, prefix=child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            lines.extend(_flatten_fact(child, prefix=f"{prefix}[{index}]"))
    elif value is not None:
        rendered = str(value).strip()
        if rendered:
            lines.append(f"{prefix}: {rendered}" if prefix else rendered)
    return lines


@dataclass(slots=True)
class CorpusDocument:
    id: str
    split: str
    language: str
    domain: str
    source_type: str
    source_id: str
    text: str
    license_name: str | None
    content_sha256: str
    approximate_tokens: int
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ProjectCorpusBuilder:
    def __init__(
        self,
        *,
        validation_fraction: float = 0.05,
        seed: int = 42,
        exclude_eval_entities: bool = True,
    ) -> None:
        if not 0.0 <= validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        self.validation_fraction = float(validation_fraction)
        self.seed = int(seed)
        self.exclude_eval_entities = bool(exclude_eval_entities)
        self.documents: list[CorpusDocument] = []
        self._content_hashes: set[str] = set()
        self._eval_questions: set[str] = set()
        self._eval_entities: set[str] = set()
        self.rejections: list[dict[str, Any]] = []

    def add_evaluation_records(self, paths: Iterable[str | Path]) -> None:
        for path in paths:
            for record in read_jsonl(path):
                question = str(record.get("question") or "").strip()
                if question:
                    self._eval_questions.add(normalize_key(question))
                for entity in flatten_entity_values(record.get("entities")):
                    normalized = normalize_key(entity)
                    if normalized:
                        self._eval_entities.add(normalized)

    def _reject(self, source_id: str, reason: str) -> None:
        self.rejections.append({"source_id": source_id, "reason": reason})

    def add_document(
        self,
        *,
        source_id: str,
        text: str,
        domain: str,
        source_type: str,
        language: str | None = None,
        license_name: str | None = None,
        entity_name: str | None = None,
        entity_aliases: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        source_id = str(source_id).strip()
        text = "\n".join(line.rstrip() for line in str(text).splitlines()).strip()
        if not source_id or not text:
            self._reject(source_id or "unknown", "empty_source_or_text")
            return
        if self.exclude_eval_entities:
            candidate_entities = [
                value
                for value in [entity_name, *(entity_aliases or [])]
                if value is not None and str(value).strip()
            ]
            if any(
                normalize_key(str(value)) in self._eval_entities
                for value in candidate_entities
                if normalize_key(str(value))
            ):
                self._reject(source_id, "held_out_entity")
                return

        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if content_hash in self._content_hashes:
            self._reject(source_id, "duplicate_text")
            return
        self._content_hashes.add(content_hash)

        document = CorpusDocument(
            id=f"corpus:{hashlib.sha256(source_id.encode('utf-8')).hexdigest()[:20]}",
            split=stable_split(
                source_id,
                validation_fraction=self.validation_fraction,
                seed=self.seed,
            ),
            language=str(language or infer_language(text)),
            domain=str(domain),
            source_type=str(source_type),
            source_id=source_id,
            text=text,
            license_name=license_name,
            content_sha256=content_hash,
            approximate_tokens=max(1, (len(text) + 3) // 4),
            metadata=dict(metadata or {}),
        )
        self.documents.append(document)

    def add_chat_jsonl(
        self,
        path: str | Path,
        *,
        domain: str,
        source_type: str = "training_conversation",
        license_name: str | None = None,
    ) -> None:
        for index, record in enumerate(read_jsonl(path), start=1):
            split = str(record.get("split") or "train").casefold()
            if split != "train":
                continue
            messages = record.get("messages")
            if not isinstance(messages, list):
                continue
            user_questions = [
                str(message.get("content") or "").strip()
                for message in messages
                if isinstance(message, Mapping) and message.get("role") == "user"
            ]
            if any(normalize_key(question) in self._eval_questions for question in user_questions):
                self._reject(str(record.get("id") or f"{path}:{index}"), "exact_eval_question")
                continue
            text = "\n\n".join(
                f"{str(message.get('role')).upper()}: {str(message.get('content')).strip()}"
                for message in messages
                if isinstance(message, Mapping) and str(message.get("content") or "").strip()
            )
            self.add_document(
                source_id=str(record.get("id") or f"{Path(path).name}:{index}"),
                text=text,
                domain=domain,
                source_type=source_type,
                language=str(record.get("language") or infer_language(text)),
                license_name=license_name,
                metadata={"path": str(path), "category": record.get("category")},
            )

    def add_guide_pages(self, path: str | Path, *, domain: str) -> None:
        for index, record in enumerate(read_jsonl(path), start=1):
            html = str(record.get("html") or "")
            text = BeautifulSoup(html, "html.parser").get_text("\n", strip=True)
            title = str(record.get("title") or record.get("requested_title") or f"page_{index}")
            license_value = record.get("license")
            license_name = (
                str(license_value.get("name"))
                if isinstance(license_value, Mapping) and license_value.get("name")
                else None
            )
            self.add_document(
                source_id=f"{domain}:guide:{title}",
                text=f"TITLE: {title}\n\n{text}",
                domain=domain,
                source_type="guide",
                language=str(record.get("language") or infer_language(text)),
                license_name=license_name,
                entity_name=title,
                metadata={
                    "source_url": record.get("source_url"),
                    "quality_status": record.get("quality_status"),
                },
            )

    def add_catalog_jsonl(
        self,
        path: str | Path,
        *,
        domain: str,
        add_bilingual_alias_bridges: bool = True,
    ) -> None:
        for index, record in enumerate(read_jsonl(path), start=1):
            name = str(
                record.get("name")
                or record.get("entity_name")
                or record.get("ItemName")
                or record.get("NPCName")
                or record.get("id")
                or f"record_{index}"
            )
            aliases = [
                str(alias).strip()
                for alias in record.get("aliases") or []
                if str(alias).strip()
            ]
            facts = record.get("facts", record)
            lines = [f"ENTITY: {name}"]
            if aliases:
                lines.append("ALIASES: " + ", ".join(aliases))
            lines.extend(_flatten_fact(facts))
            provenance = record.get("provenance")
            license_name = (
                str(provenance.get("license_name"))
                if isinstance(provenance, Mapping) and provenance.get("license_name")
                else None
            )
            source_id = str(
                record.get("source_catalog_id")
                or record.get("id")
                or f"{Path(path).name}:{index}"
            )
            self.add_document(
                source_id=f"{domain}:catalog:{source_id}",
                text="\n".join(lines),
                domain=domain,
                source_type="structured_catalog",
                language=infer_language("\n".join(lines)),
                license_name=license_name,
                entity_name=name,
                entity_aliases=aliases,
                metadata={"path": str(path), "record_type": record.get("record_type")},
            )

            # Tracked Chinese aliases provide a small but auditable bilingual
            # bridge corpus.  These documents state only the alias relation and
            # never synthesize unsupported gameplay facts.
            if add_bilingual_alias_bridges:
                chinese_aliases = [
                    alias for alias in aliases if infer_language(alias) == "zh"
                ]
                for alias in chinese_aliases:
                    self.add_document(
                        source_id=f"{domain}:alias_bridge:{source_id}:{alias}",
                        text=(
                            f"游戏实体中文别名：{alias}。"
                            f"对应的英文标准名称：{name}。"
                            f"实体类型：{record.get('record_type') or 'unknown'}。"
                        ),
                        domain=domain,
                        source_type="bilingual_alias_bridge",
                        language="zh",
                        license_name=license_name,
                        entity_name=name,
                        entity_aliases=[alias],
                        metadata={
                            "path": str(path),
                            "record_type": record.get("record_type"),
                            "source_catalog_id": source_id,
                            "generation_method": "deterministic_alias_bridge",
                        },
                    )

    def write(self, output_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
        ordered = sorted(self.documents, key=lambda item: (item.split, item.domain, item.source_id))
        write_jsonl(output_path, [document.to_dict() for document in ordered])
        distribution = {
            "split": Counter(document.split for document in ordered),
            "language": Counter(document.language for document in ordered),
            "domain": Counter(document.domain for document in ordered),
            "source_type": Counter(document.source_type for document in ordered),
        }
        manifest = {
            "schema_version": 1,
            "output": str(Path(output_path).resolve()),
            "documents": len(ordered),
            "unique_content_hashes": len(self._content_hashes),
            "approximate_tokens": sum(document.approximate_tokens for document in ordered),
            "validation_fraction": self.validation_fraction,
            "seed": self.seed,
            "exclude_eval_entities": self.exclude_eval_entities,
            "held_out_exact_questions": len(self._eval_questions),
            "held_out_entities": len(self._eval_entities),
            "distribution": {
                name: dict(sorted(counter.items())) for name, counter in distribution.items()
            },
            "rejections": {
                "count": len(self.rejections),
                "reasons": dict(sorted(Counter(item["reason"] for item in self.rejections).items())),
            },
            "claim_boundary": (
                "This is lightweight project-local causal pretraining, not full-scale "
                "foundation-model pretraining. Formal evaluation questions are excluded; "
                "entity exclusion is applied where evaluation records expose entities."
            ),
        }
        write_json(manifest_path, manifest)
        return manifest
