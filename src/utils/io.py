"""Small, dependency-light file I/O helpers used across the project."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml


def ensure_parent(path: str | Path) -> Path:
    """Create the parent directory for *path* and return the resolved path."""

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON document."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(
    path: str | Path,
    value: Any,
    *,
    indent: int | None = 2,
    sort_keys: bool = False,
    atomic: bool = True,
) -> Path:
    """Write a UTF-8 JSON document, atomically by default."""

    target = ensure_parent(path)
    text = json.dumps(
        value,
        ensure_ascii=False,
        indent=indent,
        sort_keys=sort_keys,
    ) + "\n"
    _write_text(target, text, atomic=atomic)
    return target


def read_yaml(path: str | Path) -> Any:
    """Read a UTF-8 YAML document using ``yaml.safe_load``."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_yaml(
    path: str | Path,
    value: Any,
    *,
    atomic: bool = True,
) -> Path:
    """Write a UTF-8 YAML document using ``yaml.safe_dump``."""

    target = ensure_parent(path)
    text = yaml.safe_dump(
        value,
        allow_unicode=True,
        sort_keys=False,
    )
    _write_text(target, text, atomic=atomic)
    return target


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield object records from a JSONL file with line-aware errors."""

    source = Path(path).expanduser()
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {source}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(value, dict):
                raise TypeError(
                    f"Expected a JSON object at {source}:{line_number}, "
                    f"got {type(value).__name__}."
                )
            yield value


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Return all object records from a JSONL file."""

    return list(iter_jsonl(path))


def write_jsonl(
    path: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    atomic: bool = True,
) -> Path:
    """Write object records as UTF-8 JSONL, atomically by default."""

    target = ensure_parent(path)
    lines: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise TypeError(
                f"JSONL record {index} must be a dictionary, "
                f"got {type(record).__name__}."
            )
        lines.append(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
    text = "\n".join(lines)
    if lines:
        text += "\n"
    _write_text(target, text, atomic=atomic)
    return target


def append_jsonl(path: str | Path, record: dict[str, Any]) -> Path:
    """Append one object record to a JSONL file."""

    if not isinstance(record, dict):
        raise TypeError("record must be a dictionary.")
    target = ensure_parent(path)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        handle.write("\n")
    return target


def _write_text(path: Path, text: str, *, atomic: bool) -> None:
    if not atomic:
        path.write_text(text, encoding="utf-8")
        return

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_path.replace(path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
