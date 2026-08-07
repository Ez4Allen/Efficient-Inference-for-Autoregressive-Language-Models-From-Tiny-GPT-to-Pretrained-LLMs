"""Stardew-specific retrieval over the shared guide SQLite schema."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any

from .guide_pipeline import DEFAULT_DATABASE_PATH
from .guide_query_expansion import plan_stardew_query


class StardewGuideStore:
    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        if not self.database_path.exists():
            raise FileNotFoundError(
                "Stardew guide database not found. Run `python scripts/build_stardew_guides.py`."
            )
        self.connection = sqlite3.connect(self.database_path.as_uri() + "?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row
        self._closed = False

    def __enter__(self) -> "StardewGuideStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    @staticmethod
    def _fts_query(terms: tuple[str, ...]) -> str:
        return " OR ".join(f'"{term.replace(chr(34), chr(34)*2)}"' for term in terms if term)

    @staticmethod
    def _term_set(value: str) -> set[str]:
        return set(re.findall(r"[a-z0-9][a-z0-9'_-]*", value.casefold()))

    def counts(self) -> dict[str, int]:
        return {
            "documents": int(self.connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]),
            "chunks": int(self.connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]),
        }

    def search(
        self,
        query: str,
        *,
        limit: int = 6,
        candidate_multiplier: int = 10,
        minimum_score: float = 0.17,
    ) -> list[dict[str, Any]]:
        plan = plan_stardew_query(query)
        if not plan.terms:
            return []
        limit = max(1, min(int(limit), 30))
        rows = self.connection.execute(
            """
            SELECT c.*, bm25(chunks_fts, 0.0, 3.0, 2.2, 1.8, 1.0) AS lexical_rank
            FROM chunks_fts
            JOIN chunks AS c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY lexical_rank, c.discovery_priority, c.page_title, c.position
            LIMIT ?
            """,
            (self._fts_query(plan.terms), limit * max(1, int(candidate_multiplier))),
        ).fetchall()
        query_terms = set(plan.terms)
        preferred = {title.casefold() for title in plan.preferred_titles}
        scored: list[dict[str, Any]] = []
        for ordinal, row in enumerate(rows, start=1):
            section_path = json.loads(row["section_path_json"])
            title = row["page_title"]
            combined = " ".join([title, row["section_title"], *section_path, row["text"]])
            terms = self._term_set(combined)
            coverage = len(query_terms & terms) / max(1, len(query_terms))
            title_bonus = 0.24 if title.casefold() in preferred else 0.0
            section = " ".join(section_path).casefold()
            profile_bonus = 0.0
            if plan.profile == "early_game" and any(term in section for term in ("getting started", "first", "spring", "beginning")):
                profile_bonus += 0.20
            elif plan.profile == "community_center" and any(term in section for term in ("bundle", "community center", "room")):
                profile_bonus += 0.22
            elif plan.profile == "fishing" and any(term in section for term in ("fish", "fishing", "location", "season")):
                profile_bonus += 0.18
            elif plan.profile == "skull_cavern" and (title.casefold() == "skull cavern" or any(term in section for term in ("skull", "cavern", "descent", "preparation"))):
                profile_bonus += 0.24
            elif plan.profile == "mining" and any(term in section for term in ("mine", "mining", "floor", "combat")):
                profile_bonus += 0.18
            elif plan.profile == "cooking" and (title.casefold() == "cooking" or any(term in section for term in ("recipe", "television", "friendship", "unlock"))):
                profile_bonus += 0.22
            elif plan.profile == "relationships" and any(term in section for term in ("gift", "friendship", "marriage")):
                profile_bonus += 0.18
            role_bonus = {"guide": 0.12, "mechanics": 0.08, "reference": 0.02}.get(row["retrieval_role"], 0.04)
            table_penalty = min(0.18, float(row["table_density"] or 0.0) * 0.22)
            rank_bonus = 0.08 / math.sqrt(ordinal)
            score = max(0.0, min(1.0, 0.42 * coverage + title_bonus + profile_bonus + role_bonus + rank_bonus - table_penalty))
            if score < float(minimum_score):
                continue
            scored.append(
                {
                    "chunk_id": row["chunk_id"],
                    "document_id": row["document_id"],
                    "page_title": title,
                    "section_title": row["section_title"],
                    "section_path": section_path,
                    "text": row["text"],
                    "source_url": row["source_url"],
                    "revision_id": row["revision_id"],
                    "quality_status": row["quality_status"],
                    "quality_flags": json.loads(row["quality_flags_json"]),
                    "retrieval_role": row["retrieval_role"],
                    "content_kind": row["content_kind"],
                    "table_density": float(row["table_density"] or 0.0),
                    "score": round(score, 6),
                    "rank": ordinal,
                    "matched_terms": sorted(query_terms & terms),
                    "citation_label": f"{title} — {' > '.join(section_path) if section_path else row['section_title']}",
                }
            )
        scored.sort(key=lambda item: (-item["score"], item["rank"]))
        selected: list[dict[str, Any]] = []
        section_counts: dict[str, int] = {}
        for hit in scored:
            key = f"{hit['page_title']}::{'>'.join(hit['section_path'])}"
            if section_counts.get(key, 0) >= 2:
                continue
            selected.append(hit)
            section_counts[key] = section_counts.get(key, 0) + 1
            if len(selected) >= limit:
                break
        return selected
