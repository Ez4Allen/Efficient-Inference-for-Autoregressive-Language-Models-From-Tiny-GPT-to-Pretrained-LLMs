from __future__ import annotations

from pathlib import Path

import pytest

from src.games.stardew.database_builder import DEFAULT_FACTS_PATH, build_stardew_database


@pytest.fixture()
def stardew_database(tmp_path: Path) -> Path:
    path = tmp_path / "stardew.sqlite3"
    build_stardew_database(
        facts_path=DEFAULT_FACTS_PATH,
        database_path=path,
        report_path=tmp_path / "report.json",
    )
    return path
