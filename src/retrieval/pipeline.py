"""End-to-end Terraria guide corpus build pipeline."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.utils.io import write_json
from src.utils.paths import TERRARIA_DATA_ROOT, portable_path

from .guide_database import build_guide_database
from .quality_audit import audit_guide_corpus
from .text_chunker import chunk_guide_documents
from .wiki_cleaner import clean_wiki_pages
from .wiki_importer import import_wiki_pages, utc_now


DEFAULT_GUIDES_ROOT = TERRARIA_DATA_ROOT / "guides"
DEFAULT_MANIFEST_PATH = DEFAULT_GUIDES_ROOT / "config" / "sources.yaml"


def build_terraria_guides(
    *,
    guides_root: str | Path = DEFAULT_GUIDES_ROOT,
    manifest_path: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
    max_pages: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Import, clean, chunk, audit, and index Terraria guide text."""

    guides_root = Path(guides_root).expanduser().resolve()
    manifest_path = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else guides_root / "config" / "sources.yaml"
    )
    raw_path = guides_root / "raw" / "pages.jsonl"
    documents_path = guides_root / "cleaned" / "documents.jsonl"
    chunks_path = guides_root / "chunks" / "chunks.jsonl"
    reports_root = guides_root / "reports"
    database_path = guides_root / "terraria_guides.sqlite3"
    build_report_path = reports_root / "build_report.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Guide source manifest not found: {manifest_path}")
    if offline and not raw_path.exists():
        raise FileNotFoundError(
            "Offline guide build requires an existing raw/pages.jsonl snapshot."
        )

    reports_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    stage_timings: dict[str, float] = {}

    if offline:
        import_report: dict[str, Any] = {
            "status": "skipped_offline",
            "output_path": portable_path(raw_path),
            "written_pages": None,
        }
    else:
        stage = time.perf_counter()
        import_report = import_wiki_pages(
            manifest_path=manifest_path,
            output_path=raw_path,
            report_path=reports_root / "import_report.json",
            refresh=refresh,
            max_pages=max_pages,
            verbose=verbose,
        )
        stage_timings["import"] = round(time.perf_counter() - stage, 4)
        if import_report.get("written_pages", 0) < 1:
            raise RuntimeError("Guide import produced no pages.")

    stage = time.perf_counter()
    cleaning_report = clean_wiki_pages(
        input_path=raw_path,
        output_path=documents_path,
        report_path=reports_root / "cleaning_report.json",
        manifest_path=manifest_path,
    )
    stage_timings["cleaning"] = round(time.perf_counter() - stage, 4)
    if cleaning_report.get("written_documents", 0) < 1:
        raise RuntimeError("Guide cleaning produced no documents.")

    stage = time.perf_counter()
    chunking_report = chunk_guide_documents(
        input_path=documents_path,
        output_path=chunks_path,
        report_path=reports_root / "chunking_report.json",
        manifest_path=manifest_path,
    )
    stage_timings["chunking"] = round(time.perf_counter() - stage, 4)
    if chunking_report.get("written_chunks", 0) < 1:
        raise RuntimeError("Guide chunking produced no chunks.")

    stage = time.perf_counter()
    quality_report = audit_guide_corpus(
        documents_path=documents_path,
        chunks_path=chunks_path,
        report_path=reports_root / "quality_report.json",
    )
    stage_timings["quality_audit"] = round(time.perf_counter() - stage, 4)

    stage = time.perf_counter()
    index_report = build_guide_database(
        documents_path=documents_path,
        chunks_path=chunks_path,
        database_path=database_path,
        report_path=reports_root / "index_report.json",
    )
    stage_timings["index"] = round(time.perf_counter() - stage, 4)

    total_seconds = round(time.perf_counter() - started, 4)
    report = {
        "status": "passed",
        "guides_root": portable_path(guides_root),
        "manifest_path": portable_path(manifest_path),
        "offline": bool(offline),
        "refresh": bool(refresh),
        "max_pages": max_pages,
        "outputs": {
            "raw_pages": portable_path(raw_path),
            "documents": portable_path(documents_path),
            "chunks": portable_path(chunks_path),
            "database": portable_path(database_path),
        },
        "counts": {
            "pages": (
                import_report.get("written_pages")
                if not offline
                else cleaning_report.get("input_pages")
            ),
            "documents": cleaning_report.get("written_documents"),
            "sections": cleaning_report.get("section_count"),
            "chunks": chunking_report.get("written_chunks"),
        },
        "quality": {
            "import_status": import_report.get("status"),
            "cleaning_status": cleaning_report.get("status"),
            "index_integrity": index_report.get("integrity_check"),
            "import_errors": len(import_report.get("errors") or []),
            "skipped_documents": cleaning_report.get("skipped_documents"),
            "duplicate_chunks_skipped": chunking_report.get(
                "duplicate_chunks_skipped"
            ),
            "quality_audit_status": quality_report.get("status"),
            "quality_warning_counts": quality_report.get("warning_counts"),
        },
        "timings_seconds": {**stage_timings, "total": total_seconds},
        "generated_at": utc_now(),
    }
    write_json(build_report_path, report)
    if verbose:
        print(
            "Terraria guide build passed: "
            f"{report['counts']['documents']} documents, "
            f"{report['counts']['chunks']} chunks, "
            f"{total_seconds:.2f}s."
        )
    return report
