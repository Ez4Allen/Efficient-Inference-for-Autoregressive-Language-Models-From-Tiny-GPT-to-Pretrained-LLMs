from __future__ import annotations

from pathlib import Path

from src.games.stardew.assistant import StardewAssistant
from src.games.stardew.guide_pipeline import build_stardew_seed_guides


def test_offline_seed_build_and_guide_answer(tmp_path: Path) -> None:
    project_guides = Path(__file__).resolve().parents[3] / "data" / "stardew" / "guides"
    root = tmp_path / "guides"
    (root / "config").mkdir(parents=True)
    (root / "seed").mkdir(parents=True)
    (root / "config" / "sources.yaml").write_text(
        (project_guides / "config" / "sources.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (root / "seed" / "pages.jsonl").write_text(
        (project_guides / "seed" / "pages.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    report = build_stardew_seed_guides(guides_root=root, verbose=False)
    assert report["status"] == "passed"
    assert report["seed"] is True
    assert report["counts"]["documents"] == 4
    assert report["counts"]["chunks"] >= 8


def test_assistant_auto_builds_seed_guide_database(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[3]
    guide_root = tmp_path / "guides"
    (guide_root / "config").mkdir(parents=True)
    (guide_root / "seed").mkdir(parents=True)
    source_root = project_root / "data" / "stardew" / "guides"
    (guide_root / "config" / "sources.yaml").write_text(
        (source_root / "config" / "sources.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (guide_root / "seed" / "pages.jsonl").write_text(
        (source_root / "seed" / "pages.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    catalog_db = tmp_path / "stardew.sqlite3"
    guide_db = guide_root / "stardew_guides.sqlite3"

    with StardewAssistant(
        database_path=catalog_db,
        guide_database_path=guide_db,
        auto_build=True,
    ) as assistant:
        result = assistant.answer("What should I prioritize during the first spring?")

    assert guide_db.exists()
    assert result.status == "found"
    assert result.intent == "guide"
    assert result.evidence
    assert any("Getting Started" in item.label for item in result.evidence)
