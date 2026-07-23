"""Repository path discovery and path-resolution helpers.

The project is primarily used from a source checkout (locally or in Colab),
so paths must not depend on a fixed ``/content/llm_project`` location.  Set
``LLM_PROJECT_ROOT`` to override automatic discovery when embedding the
package in another application.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV = "LLM_PROJECT_ROOT"


def find_project_root(start: str | Path | None = None) -> Path:
    """Return the repository root.

    Discovery order:

    1. ``LLM_PROJECT_ROOT`` environment variable.
    2. Walk upward from ``start`` (or this module) looking for a repository
       that contains both ``src`` and ``configs``.
    3. Fall back to the current working directory.
    """

    configured_root = os.environ.get(PROJECT_ROOT_ENV)
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(
                f"{PROJECT_ROOT_ENV} points to a missing directory: {root}"
            )
        return root

    candidate = Path(start) if start is not None else Path(__file__)
    candidate = candidate.expanduser().resolve()
    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        if (directory / "src").is_dir() and (directory / "configs").is_dir():
            return directory

    return Path.cwd().resolve()


def resolve_project_path(
    value: str | Path,
    *,
    base: str | Path | None = None,
) -> Path:
    """Expand environment variables and resolve a path deterministically.

    Relative paths are interpreted relative to ``base`` when provided,
    otherwise relative to :data:`PROJECT_ROOT`.
    """

    raw_value = str(value)
    expanded = os.path.expandvars(os.path.expanduser(raw_value))
    if "$" in expanded:
        raise ValueError(
            f"Path contains an unresolved environment variable: {raw_value}"
        )
    path = Path(expanded)
    if path.is_absolute():
        return path.resolve()

    base_path = Path(base) if base is not None else PROJECT_ROOT
    return (base_path / path).resolve()


PROJECT_ROOT = find_project_root()
DATA_ROOT = PROJECT_ROOT / "data"
RESULTS_ROOT = PROJECT_ROOT / "results"
CHECKPOINTS_ROOT = PROJECT_ROOT / "checkpoints"

TERRARIA_DATA_ROOT = DATA_ROOT / "terraria"
TERRARIA_CATALOG_ROOT = TERRARIA_DATA_ROOT / "catalog"
TERRARIA_CLEANED_ROOT = TERRARIA_CATALOG_ROOT / "cleaned"
TERRARIA_LINKED_ROOT = TERRARIA_CATALOG_ROOT / "linked"


def portable_path(
    value: str | Path,
    *,
    root: str | Path | None = None,
) -> str:
    """Return a project-relative POSIX path when possible.

    Paths outside the selected root remain absolute. This keeps tracked reports
    portable while preserving unambiguous paths for temporary/test builds.
    """

    resolved = Path(value).expanduser().resolve()
    selected_root = Path(root).expanduser().resolve() if root is not None else PROJECT_ROOT
    try:
        return resolved.relative_to(selected_root).as_posix()
    except ValueError:
        return str(resolved)
