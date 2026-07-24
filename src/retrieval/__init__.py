"""Document ingestion and retrieval for Terraria guides."""

from .guide_database import (
    DEFAULT_GUIDE_DATABASE_PATH,
    GuideDocumentStore,
    build_guide_database,
)
from .pipeline import build_terraria_guides
from .schemas import GuideChunk, GuideDocument, GuideSearchHit, GuideSection

__all__ = [
    "DEFAULT_GUIDE_DATABASE_PATH",
    "GuideChunk",
    "GuideDocument",
    "GuideDocumentStore",
    "GuideSearchHit",
    "GuideSection",
    "build_guide_database",
    "build_terraria_guides",
]
