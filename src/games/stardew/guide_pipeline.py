"""End-to-end Stardew Valley guide corpus pipeline."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from src.retrieval.guide_database import build_guide_database
from src.retrieval.quality_audit import audit_guide_corpus
from src.retrieval.text_chunker import chunk_guide_documents
from src.retrieval.wiki_cleaner import clean_wiki_pages
from src.retrieval.wiki_importer import import_wiki_pages, utc_now
from src.utils.io import write_json
from src.utils.paths import STARDEW_GUIDES_ROOT, portable_path

DEFAULT_GUIDES_ROOT = STARDEW_GUIDES_ROOT
DEFAULT_MANIFEST_PATH = DEFAULT_GUIDES_ROOT / "config" / "sources.yaml"
DEFAULT_DATABASE_PATH = DEFAULT_GUIDES_ROOT / "stardew_guides.sqlite3"
DEFAULT_SEED_PATH = DEFAULT_GUIDES_ROOT / "seed" / "pages.jsonl"


def build_stardew_guides(
    *,
    guides_root: str | Path = DEFAULT_GUIDES_ROOT,
    manifest_path: str | Path | None = None,
    offline: bool = False,
    refresh: bool = False,
    max_pages: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    guides_root = Path(guides_root).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path else guides_root / "config" / "sources.yaml"
    raw_path = guides_root / "raw" / "pages.jsonl"
    documents_path = guides_root / "cleaned" / "documents.jsonl"
    chunks_path = guides_root / "chunks" / "chunks.jsonl"
    reports_root = guides_root / "reports"
    database_path = guides_root / "stardew_guides.sqlite3"
    build_report_path = reports_root / "build_report.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Stardew guide manifest not found: {manifest_path}")
    if offline and not raw_path.exists():
        raise FileNotFoundError("Offline Stardew guide build requires raw/pages.jsonl.")
    reports_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    if offline:
        import_report: dict[str, Any] = {"status": "skipped_offline", "written_pages": None}
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
        timings["import"] = round(time.perf_counter() - stage, 4)
        if import_report.get("written_pages", 0) < 1:
            raise RuntimeError("Stardew guide import produced no pages.")
    stage = time.perf_counter()
    cleaning = clean_wiki_pages(
        input_path=raw_path,
        output_path=documents_path,
        report_path=reports_root / "cleaning_report.json",
        manifest_path=manifest_path,
    )
    timings["cleaning"] = round(time.perf_counter() - stage, 4)
    stage = time.perf_counter()
    chunking = chunk_guide_documents(
        input_path=documents_path,
        output_path=chunks_path,
        report_path=reports_root / "chunking_report.json",
        manifest_path=manifest_path,
    )
    timings["chunking"] = round(time.perf_counter() - stage, 4)
    stage = time.perf_counter()
    quality = audit_guide_corpus(
        documents_path=documents_path,
        chunks_path=chunks_path,
        report_path=reports_root / "quality_report.json",
    )
    timings["quality_audit"] = round(time.perf_counter() - stage, 4)
    stage = time.perf_counter()
    index = build_guide_database(
        documents_path=documents_path,
        chunks_path=chunks_path,
        database_path=database_path,
        report_path=reports_root / "index_report.json",
    )
    timings["index"] = round(time.perf_counter() - stage, 4)
    total = round(time.perf_counter() - started, 4)
    report = {
        "status": "passed",
        "game": "stardew_valley",
        "guides_root": portable_path(guides_root),
        "manifest_path": portable_path(manifest_path),
        "offline": bool(offline),
        "refresh": bool(refresh),
        "max_pages": max_pages,
        "counts": {
            "pages": import_report.get("written_pages") if not offline else cleaning.get("input_pages"),
            "documents": cleaning.get("written_documents"),
            "sections": cleaning.get("section_count"),
            "chunks": chunking.get("written_chunks"),
        },
        "quality": {
            "import_status": import_report.get("status"),
            "import_errors": len(import_report.get("errors") or []),
            "cleaning_status": cleaning.get("status"),
            "quality_audit_status": quality.get("status"),
            "quality_warning_counts": quality.get("warning_counts"),
            "index_integrity": index.get("integrity_check"),
        },
        "outputs": {
            "raw_pages": portable_path(raw_path),
            "documents": portable_path(documents_path),
            "chunks": portable_path(chunks_path),
            "database": portable_path(database_path),
        },
        "timings_seconds": {**timings, "total": total},
        "generated_at": utc_now(),
    }
    write_json(build_report_path, report)
    if verbose:
        print(
            f"Stardew guide build passed: {report['counts']['documents']} documents, "
            f"{report['counts']['chunks']} chunks, {total:.2f}s."
        )
    return report

def build_stardew_seed_guides(
    *,
    guides_root: str | Path = DEFAULT_GUIDES_ROOT,
    manifest_path: str | Path | None = None,
    seed_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Build a small offline guide index from the tracked curated seed.

    The seed is a source-attributed, project-authored set of summaries intended
    for deterministic demos and release tests. A normal online build replaces
    the generated raw snapshot with the configured Wiki pages.
    """

    guides_root = Path(guides_root).expanduser().resolve()
    manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path
        else guides_root / "config" / "sources.yaml"
    )
    seed = (
        Path(seed_path).expanduser().resolve()
        if seed_path
        else guides_root / "seed" / "pages.jsonl"
    )
    if not seed.exists():
        raise FileNotFoundError(f"Stardew guide seed not found: {seed}")
    raw_path = guides_root / "raw" / "pages.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(seed, raw_path)
    report = build_stardew_guides(
        guides_root=guides_root,
        manifest_path=manifest,
        offline=True,
        verbose=verbose,
    )
    report["seed"] = True
    report["seed_path"] = portable_path(seed)
    write_json(guides_root / "reports" / "build_report.json", report)
    return report
