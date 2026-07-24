"""SQLite FTS5 index and read-only store for Terraria guide chunks."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl, write_json
from src.utils.paths import PROJECT_ROOT, TERRARIA_DATA_ROOT, portable_path

from .query_expansion import plan_guide_query
from .schemas import GuideSearchHit
from .wiki_importer import utc_now


DEFAULT_GUIDES_ROOT = TERRARIA_DATA_ROOT / "guides"
DEFAULT_GUIDE_DATABASE_PATH = DEFAULT_GUIDES_ROOT / "terraria_guides.sqlite3"
DEFAULT_GUIDE_DATABASE_REPORT_PATH = DEFAULT_GUIDES_ROOT / "reports" / "index_report.json"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _deduplicate_records(
    records: list[dict[str, Any]],
    *,
    id_field: str,
    record_kind: str,
) -> tuple[list[dict[str, Any]], int]:
    """
    Deduplicate records by their stable identifier.

    Exact duplicates are collapsed deterministically.
    Conflicting duplicates with the same ID but different
    content hashes are rejected instead of silently overwritten.
    """

    unique: dict[str, dict[str, Any]] = {}
    duplicate_count = 0

    for record in records:
        record_id = str(
            record.get(id_field, "")
        ).strip()

        if not record_id:
            raise ValueError(
                f"{record_kind} record is missing "
                f"required field {id_field!r}."
            )

        existing = unique.get(record_id)

        if existing is None:
            unique[record_id] = record
            continue

        duplicate_count += 1

        existing_hash = existing.get(
            "content_sha256"
        )

        incoming_hash = record.get(
            "content_sha256"
        )

        if (
            existing_hash
            and incoming_hash
            and existing_hash != incoming_hash
        ):
            raise ValueError(
                f"Conflicting duplicate {record_kind} ID: "
                f"{record_id}\n"
                f"existing content_sha256={existing_hash}\n"
                f"incoming content_sha256={incoming_hash}"
            )

        # When hashes agree, keep the first occurrence.
        # Discovery order is deterministic and the records
        # represent the same source content.

    return list(unique.values()), duplicate_count


def build_guide_database(
    *,
    documents_path: str | Path,
    chunks_path: str | Path,
    database_path: str | Path = DEFAULT_GUIDE_DATABASE_PATH,
    report_path: str | Path = DEFAULT_GUIDE_DATABASE_REPORT_PATH,
) -> dict[str, Any]:
    """Build a normalized SQLite/FTS database from cleaned documents and chunks."""

    documents_path = Path(documents_path).expanduser().resolve()
    chunks_path = Path(chunks_path).expanduser().resolve()
    database_path = Path(database_path).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve()
    documents = read_jsonl(documents_path)
    chunks = read_jsonl(chunks_path)

    if not documents or not chunks:
        raise ValueError(
            "Guide database requires at least one "
            "document and chunk."
        )

    documents, duplicate_document_count = (
        _deduplicate_records(
            documents,
            id_field="document_id",
            record_kind="document",
        )
    )

    chunks, duplicate_chunk_count = (
        _deduplicate_records(
            chunks,
            id_field="chunk_id",
            record_kind="chunk",
        )
    )

    valid_document_ids = {
        row["document_id"]
        for row in documents
    }

    orphan_chunk_ids = [
        row["chunk_id"]
        for row in chunks
        if row["document_id"]
        not in valid_document_ids
    ]

    if orphan_chunk_ids:
        raise ValueError(
            "Guide chunks reference missing documents. "
            f"Examples: {orphan_chunk_ids[:10]}"
        )

    database_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = database_path.with_suffix(database_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    connection = sqlite3.connect(temporary)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = OFF")
        connection.execute("PRAGMA synchronous = OFF")
        connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE documents (
                document_id TEXT PRIMARY KEY,
                page_title TEXT NOT NULL,
                normalized_title TEXT NOT NULL,
                page_id INTEGER,
                revision_id INTEGER,
                revision_timestamp TEXT,
                source_url TEXT NOT NULL,
                language TEXT NOT NULL,
                quality_status TEXT NOT NULL,
                quality_flags_json TEXT NOT NULL,
                content_sha256 TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                retrieval_role TEXT NOT NULL,
                discovery_priority INTEGER NOT NULL,
                parse_status TEXT NOT NULL,
                record_json TEXT NOT NULL
            ) WITHOUT ROWID;

            CREATE TABLE chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                page_title TEXT NOT NULL,
                normalized_title TEXT NOT NULL,
                section_title TEXT NOT NULL,
                section_path_json TEXT NOT NULL,
                text TEXT NOT NULL,
                source_url TEXT NOT NULL,
                revision_id INTEGER,
                language TEXT NOT NULL,
                quality_status TEXT NOT NULL,
                quality_flags_json TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                retrieval_role TEXT NOT NULL,
                discovery_priority INTEGER NOT NULL,
                content_kind TEXT NOT NULL,
                table_row_count INTEGER NOT NULL,
                table_density REAL NOT NULL,
                content_sha256 TEXT NOT NULL,
                character_count INTEGER NOT NULL,
                word_count INTEGER NOT NULL,
                FOREIGN KEY(document_id) REFERENCES documents(document_id)
            ) WITHOUT ROWID;

            CREATE INDEX idx_documents_title ON documents(normalized_title);
            CREATE INDEX idx_documents_quality ON documents(quality_status);
            CREATE INDEX idx_documents_role ON documents(retrieval_role, discovery_priority);
            CREATE INDEX idx_chunks_document ON chunks(document_id, position);
            CREATE INDEX idx_chunks_title ON chunks(normalized_title);
            CREATE INDEX idx_chunks_quality ON chunks(quality_status);
            CREATE INDEX idx_chunks_role ON chunks(retrieval_role, discovery_priority);
            CREATE INDEX idx_chunks_kind ON chunks(content_kind, table_density);

            CREATE VIRTUAL TABLE chunks_fts USING fts5(
                chunk_id UNINDEXED,
                page_title,
                section_title,
                section_path,
                text,
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [
                ("schema_version", "1"),
                ("database_type", "terraria_guide_corpus"),
                ("generated_at", utc_now()),
                ("documents_path", portable_path(documents_path)),
                ("chunks_path", portable_path(chunks_path)),
            ],
        )
        connection.executemany(
            """
            INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["document_id"],
                    row["page_title"],
                    row["normalized_title"],
                    row.get("page_id"),
                    row.get("revision_id"),
                    row.get("revision_timestamp"),
                    row["source_url"],
                    row.get("language", "en"),
                    row.get("quality_status", "unknown"),
                    _json(row.get("quality_flags") or []),
                    row["content_sha256"],
                    row.get("source_kind", "unknown"),
                    row.get("retrieval_role", "guide"),
                    int(row.get("discovery_priority", 10_000)),
                    row.get("parse_status", "ok"),
                    _json(row),
                )
                for row in documents
            ],
        )
        chunk_rows = [
            (
                row["chunk_id"],
                row["document_id"],
                int(row["position"]),
                row["page_title"],
                row["normalized_title"],
                row["section_title"],
                _json(row.get("section_path") or []),
                row["text"],
                row["source_url"],
                row.get("revision_id"),
                row.get("language", "en"),
                row.get("quality_status", "unknown"),
                _json(row.get("quality_flags") or []),
                row.get("source_kind", "unknown"),
                row.get("retrieval_role", "guide"),
                int(row.get("discovery_priority", 10_000)),
                row.get("content_kind", "prose"),
                int(row.get("table_row_count", 0)),
                float(row.get("table_density", 0.0)),
                row["content_sha256"],
                int(row["character_count"]),
                int(row["word_count"]),
            )
            for row in chunks
        ]
        connection.executemany(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            chunk_rows,
        )
        connection.executemany(
            "INSERT INTO chunks_fts VALUES (?, ?, ?, ?, ?)",
            [
                (
                    row["chunk_id"],
                    row["page_title"],
                    row["section_title"],
                    " > ".join(row.get("section_path") or []),
                    row["text"],
                )
                for row in chunks
            ],
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        foreign_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
        if integrity is None or integrity[0] != "ok":
            raise AssertionError(f"SQLite integrity check failed: {integrity}")
        if foreign_errors:
            raise AssertionError(f"Guide database foreign-key errors: {foreign_errors[:5]}")
    except Exception:
        connection.close()
        temporary.unlink(missing_ok=True)
        raise
    else:
        connection.close()

    temporary.replace(database_path)
    report = {
        "status": "passed",
        "documents_path": portable_path(documents_path),
        "chunks_path": portable_path(chunks_path),
        "database_path": portable_path(database_path),
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "duplicate_documents_skipped": (
            duplicate_document_count
        ),
        "duplicate_chunks_skipped": (
            duplicate_chunk_count
        ),
        "quality_status_counts": dict(
            Counter(
                row.get(
                    "quality_status",
                    "unknown",
                )
                for row in chunks
            )
        ),
        "fts_enabled": True,
        "integrity_check": "ok",
        "foreign_key_errors": 0,
        "database_size_bytes": database_path.stat().st_size,
        "generated_at": utc_now(),
    }
    write_json(report_path, report)
    return report


class GuideDocumentStore:
    """Read-only lexical retrieval over the local guide corpus."""

    QUALITY_BONUS = {
        "revised": 1.0,
        "unknown": 0.82,
        "under_revision": 0.62,
        "subject_to_revision": 0.45,
        "legacy": 0.15,
    }

    def __init__(self, database_path: str | Path = DEFAULT_GUIDE_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        if not self.database_path.exists():
            raise FileNotFoundError(
                "Terraria guide database not found. Build it with "
                "`python scripts/build_terraria_guides.py`. "
                f"Expected path: {self.database_path}"
            )
        uri = self.database_path.as_uri() + "?mode=ro"
        self.connection = sqlite3.connect(uri, uri=True)
        self.connection.row_factory = sqlite3.Row
        self._closed = False

    def __enter__(self) -> "GuideDocumentStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("GuideDocumentStore is closed.")

    def counts(self) -> dict[str, int]:
        self._ensure_open()
        return {
            "documents": int(self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]),
            "chunks": int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
        }

    @staticmethod
    def _fts_query(terms: list[str]) -> str:
        safe_terms = []
        for term in terms:
            cleaned = str(term).replace('"', '""').strip()
            if cleaned:
                safe_terms.append(f'"{cleaned}"')
        return " OR ".join(safe_terms)

    @staticmethod
    def _term_set(value: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", value)
            if token
        }

    @staticmethod
    def _normalized_text(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).strip()

    @classmethod
    def _phrase_match_score(cls, value: str, phrases: tuple[str, ...]) -> float:
        """Return a section/path match score that rewards exact phrases."""

        if not phrases:
            return 0.0
        normalized_value = cls._normalized_text(value)
        value_terms = cls._term_set(normalized_value)
        best = 0.0
        for phrase in phrases:
            normalized_phrase = cls._normalized_text(phrase)
            if not normalized_phrase:
                continue
            if normalized_phrase in normalized_value:
                best = max(best, 1.0)
                continue
            phrase_terms = cls._term_set(normalized_phrase)
            if phrase_terms:
                best = max(best, 0.65 * len(phrase_terms & value_terms) / len(phrase_terms))
        return best

    @classmethod
    def _profile_adjustment(
        cls,
        *,
        profile: str,
        page_title: str,
        section_path: list[str],
        text: str,
    ) -> tuple[float, bool]:
        """Apply intent-specific stage and scope signals.

        The boolean return value marks a hard stage mismatch. Such chunks are
        excluded instead of merely being demoted because they would produce
        misleading grounded answers (for example, Pre-Hardmode preparation for
        a question asking what to do *after* entering Hardmode).
        """

        title = cls._normalized_text(page_title.removeprefix("Guide:"))
        path = cls._normalized_text(" > ".join(section_path))
        body = cls._normalized_text(text)
        adjustment = 0.0

        # Exclude challenge-seed guides from generic advice queries.
        # These pages describe intentionally altered world rules and can
        # produce misleading answers for a normal Terraria playthrough.
        special_seed_titles = {
            "remix",
            "not the bees",
            "the constant",
            "skyblock",
            "true zenith",
        }

        if (
            profile
            in {
                "first_night",
                "early_hardmode",
                "boss_progression",
                "biome_spread",
            }
            and title in special_seed_titles
        ):
            return adjustment, True

        if profile == "first_night":
            if any(phrase in path for phrase in ("the first day", "first night", "safety and house building")):
                adjustment += 0.24
            if any(phrase in path for phrase in ("character creation", "world creation")):
                adjustment -= 0.24
            elif "exploring" in path and "night" not in path:
                adjustment -= 0.14

        elif profile == "early_hardmode":
            if any(
                phrase in path
                for phrase in (
                    "pre hardmode",
                    "preparing for hardmode",
                    "character creation",
                    "world creation",
                    "gamemode",
                )
            ):
                return adjustment, True

            early_path = any(
                phrase in path
                for phrase in ("early hardmode", "hardmode priorities")
            )
            mechanics_overview_path = title == "hardmode" and any(
                phrase in path for phrase in ("enemies", "ores and bars", "biomes")
            )
            early_body_signals = sum(
                phrase in body
                for phrase in (
                    "entering hardmode",
                    "enter hardmode",
                    "wall of flesh",
                    "early hardmode",
                    "hardmode ores",
                    "biome spread",
                    "mechanical bosses",
                )
            )
            if not early_path and not mechanics_overview_path and early_body_signals == 0:
                return adjustment, True

            if early_path:
                adjustment += 0.30
            elif path == "hardmode" or path.endswith(" hardmode"):
                adjustment += 0.10
            if mechanics_overview_path:
                adjustment += 0.06
            if any(
                phrase in path
                for phrase in (
                    "after plantera",
                    "post plantera",
                    "hardmode jungle",
                    "the frost legion",
                    "lunar events",
                    "conversion",
                    "after one mechanical boss",
                    "the second mechanical boss",
                )
            ):
                adjustment -= 0.24
            if "wall of flesh" in body and "hardmode" in body:
                adjustment += 0.06

        elif profile == "boss_progression":
            boss_markers = (
                "eye of cthulhu",
                "eater of worlds",
                "brain of cthulhu",
                "skeletron",
                "wall of flesh",
                "destroyer",
                "twins",
                "skeletron prime",
                "plantera",
                "golem",
                "lunatic cultist",
                "moon lord",
            )
            distinct_bosses = sum(marker in body for marker in boss_markers)
            broad_boss_path = path == "overview" or "boss" in path
            if not broad_boss_path and distinct_bosses < 5:
                return adjustment, True

            if title == "game progression" and path == "overview":
                adjustment += 0.34
            elif "boss progression" in path:
                adjustment += 0.30
            elif title == "bosses" and path in {"overview", "bosses"}:
                adjustment += 0.22
            elif path == "overview":
                adjustment += 0.14

            # Broad order questions should begin with a cross-game overview, not
            # a single optional boss or one narrow progression checkpoint.
            if "after the end" in path:
                return adjustment, True
            if any(
                phrase in path
                for phrase in (
                    "after plantera",
                    "late pre hardmode",
                    "mechanical bosses",
                    "the second mechanical boss",
                    "after one mechanical boss",
                )
            ):
                adjustment -= 0.18

            if distinct_bosses >= 5:
                adjustment += 0.18
            elif distinct_bosses >= 3:
                adjustment += 0.10

        elif profile == "biome_spread":
            spread_path = any(
                phrase in path
                for phrase in (
                    "biome spread",
                    "spread",
                    "world purity",
                    "containing and preventing",
                    "biomes",
                )
            )
            spread_body = "spread" in body and any(
                term in body for term in ("corruption", "crimson", "hallow")
            )
            if not spread_path and not spread_body:
                return adjustment, True
            if spread_path:
                adjustment += 0.26
            if "post plantera" in path:
                adjustment -= 0.12
            if any(term in body for term in ("corruption", "crimson", "hallow")):
                adjustment += 0.05

        return adjustment, False

    @staticmethod
    def _diversify(
        hits: list[GuideSearchHit],
        *,
        limit: int,
        profile: str,
    ) -> list[GuideSearchHit]:
        """Avoid redundant results while preserving useful continuations."""

        if profile not in {
            "boss_progression",
            "early_hardmode",
            "first_night",
        }:
            return hits[:limit]

        selected: list[GuideSearchHit] = []
        top_level_counts: Counter[str] = Counter()
        exact_section_counts: Counter[str] = Counter()

        for hit in hits:
            top_level = (
                hit.section_path[0]
                if hit.section_path
                else hit.section_title
            )

            top_key = top_level.casefold()

            exact_key = " > ".join(
                hit.section_path
                or [hit.section_title]
            ).casefold()

            is_progression_overview = (
                profile == "boss_progression"
                and hit.page_title
                == "Guide:Game progression"
                and exact_key == "overview"
            )

            # The progression Overview is split across multiple chunks.
            # Its first chunk introduces the sequence and the following
            # chunks contain the actual boss list, so retain up to three.
            exact_limit = (
                3
                if is_progression_overview
                else 1
            )

            top_level_limit = (
                3
                if is_progression_overview
                else 2
            )

            if (
                exact_section_counts[
                    exact_key
                ]
                >= exact_limit
            ):
                continue

            if (
                top_level_counts[
                    top_key
                ]
                >= top_level_limit
            ):
                continue

            selected.append(hit)

            exact_section_counts[
                exact_key
            ] += 1

            top_level_counts[
                top_key
            ] += 1

            if len(selected) >= limit:
                return selected

        return selected
    def search(
        self,
        query: str,
        *,
        limit: int = 6,
        candidate_multiplier: int = 8,
        minimum_score: float = 0.14,
        include_low_quality: bool = True,
    ) -> list[dict[str, Any]]:
        self._ensure_open()
        limit = max(1, min(int(limit), 50))
        candidate_limit = max(limit, min(limit * max(1, int(candidate_multiplier)), 400))
        plan = plan_guide_query(query)
        profile_minimums = {
            "first_night": 0.20,
            "early_hardmode": 0.22,
            "boss_progression": 0.22,
            "biome_spread": 0.22,
        }
        effective_minimum_score = max(
            float(minimum_score),
            profile_minimums.get(plan.profile, 0.0),
        )
        terms = list(plan.terms)
        if not terms:
            return []
        fts_query = self._fts_query(terms)
        quality_clause = "" if include_low_quality else (
            "AND c.quality_status NOT IN ('subject_to_revision', 'legacy')"
        )
        rows = self.connection.execute(
            f"""
            SELECT
                c.*,
                bm25(chunks_fts, 0.0, 3.0, 2.2, 1.8, 1.0) AS lexical_rank
            FROM chunks_fts
            JOIN chunks AS c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
              {quality_clause}
            ORDER BY lexical_rank, c.discovery_priority, c.page_title, c.position
            LIMIT ?
            """,
            (fts_query, candidate_limit),
        ).fetchall()

        query_terms = set(terms)
        original_terms = set(plan.original_terms)
        anchor_terms = set(plan.anchor_terms)
        preferred_titles = plan.normalized_preferred_titles
        results: list[GuideSearchHit] = []
        for ordinal, row in enumerate(rows, start=1):
            section_path = json.loads(row["section_path_json"])
            title_terms = self._term_set(row["page_title"])
            section_terms = self._term_set(row["section_title"])
            path_terms = self._term_set(" ".join(section_path))
            text_terms = self._term_set(row["text"])
            all_terms = title_terms | section_terms | path_terms | text_terms
            matched = sorted(query_terms & all_terms)
            coverage = len(matched) / max(1, len(query_terms))
            original_coverage = (
                len(original_terms & all_terms) / len(original_terms)
                if original_terms
                else 0.0
            )
            anchor_coverage = (
                len(anchor_terms & all_terms) / len(anchor_terms)
                if anchor_terms
                else 0.0
            )
            title_coverage = len(query_terms & title_terms) / max(1, len(query_terms))
            section_coverage = len(query_terms & (section_terms | path_terms)) / max(1, len(query_terms))
            full_path = " > ".join([row["page_title"], *section_path])
            preferred_path_score = self._phrase_match_score(
                full_path,
                plan.preferred_sections,
            )
            discouraged_path_score = self._phrase_match_score(
                " > ".join(section_path),
                plan.discouraged_sections,
            )
            normalized_title = self._normalized_text(
                row["page_title"].removeprefix("Guide:")
            )
            preferred_title_bonus = 1.0 if normalized_title in preferred_titles else 0.0
            rank_score = 1.0 / math.sqrt(ordinal)
            quality_bonus = self.QUALITY_BONUS.get(row["quality_status"], 0.7)
            role_bonus = {
                "guide": 1.0,
                "mechanics": 0.82,
                "reference": 0.35,
            }.get(row["retrieval_role"], 0.6)
            table_density = float(row["table_density"] or 0.0)
            table_penalty = 0.0
            if plan.advice_query:
                table_penalty = min(0.24, table_density * 0.28)
                if row["retrieval_role"] == "reference":
                    table_penalty += 0.10

            profile_adjustment, hard_mismatch = self._profile_adjustment(
                profile=plan.profile,
                page_title=row["page_title"],
                section_path=section_path,
                text=row["text"],
            )
            if hard_mismatch:
                continue

            score = (
                0.22 * coverage
                + 0.13 * original_coverage
                + 0.08 * title_coverage
                + 0.08 * section_coverage
                + 0.12 * preferred_path_score
                + 0.12 * preferred_title_bonus
                + 0.10 * anchor_coverage
                + 0.04 * rank_score
                + 0.05 * quality_bonus
                + 0.05 * role_bonus
                + profile_adjustment
                - 0.18 * discouraged_path_score
                - table_penalty
            )
            score = max(0.0, min(1.0, score))
            if score < effective_minimum_score:
                continue
            results.append(
                GuideSearchHit(
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    page_title=row["page_title"],
                    section_title=row["section_title"],
                    section_path=section_path,
                    text=row["text"],
                    source_url=row["source_url"],
                    revision_id=row["revision_id"],
                    quality_status=row["quality_status"],
                    quality_flags=json.loads(row["quality_flags_json"]),
                    score=round(score, 6),
                    rank=ordinal,
                    matched_terms=matched,
                    retrieval_role=row["retrieval_role"],
                    content_kind=row["content_kind"],
                    table_density=round(table_density, 6),
                )
            )

        results.sort(key=lambda hit: (-hit.score, hit.rank, hit.page_title))
        diversified = self._diversify(results, limit=limit, profile=plan.profile)
        return [hit.to_dict() for hit in diversified]

    def get_document(self, document_id: str) -> dict[str, Any] | None:
        self._ensure_open()
        row = self.connection.execute(
            "SELECT record_json FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
        return json.loads(row[0]) if row else None
