"""Discover and import Terraria guide pages through the MediaWiki API."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.io import read_jsonl, read_yaml, write_json, write_jsonl
from src.utils.paths import portable_path

from .schemas import WikiPageDescriptor
from .wiki_client import MediaWikiAPIError, MediaWikiClient


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_source_manifest(path: str | Path) -> dict[str, Any]:
    manifest = read_yaml(path)
    if not isinstance(manifest, dict):
        raise TypeError("Guide source manifest must contain a YAML object.")
    if int(manifest.get("version", 0)) != 1:
        raise ValueError("Unsupported guide source manifest version.")
    wiki = manifest.get("wiki")
    discovery = manifest.get("discovery")
    if not isinstance(wiki, dict) or not isinstance(discovery, dict):
        raise ValueError("Manifest requires wiki and discovery mappings.")
    for field in ("api_url", "article_base_url", "language"):
        if not wiki.get(field):
            raise ValueError(f"Manifest wiki.{field} is required.")
    return manifest


def create_client(manifest: dict[str, Any]) -> MediaWikiClient:
    wiki = manifest["wiki"]
    request = wiki.get("request") or {}
    return MediaWikiClient(
        api_url=wiki["api_url"],
        article_base_url=wiki["article_base_url"],
        user_agent=request.get(
            "user_agent",
            "EfficientInferenceTerrariaResearch/0.5",
        ),
        timeout_seconds=float(request.get("timeout_seconds", 30)),
        request_delay_seconds=float(request.get("request_delay_seconds", 0.35)),
        max_retries=int(request.get("max_retries", 4)),
    )


def _quality_priority(status: str) -> int:
    return {
        "subject_to_revision": 4,
        "under_revision": 3,
        "revised": 2,
        "unknown": 1,
    }.get(status, 0)


def discover_pages(
    client: MediaWikiClient,
    manifest: dict[str, Any],
    *,
    max_pages: int | None = None,
) -> list[WikiPageDescriptor]:
    """Discover current guide pages plus explicitly configured mechanics pages."""

    discovery = manifest["discovery"]
    include_prefixes = tuple(str(value) for value in discovery.get("include_title_prefixes", []))
    exclude_prefixes = tuple(str(value) for value in discovery.get("exclude_title_prefixes", []))
    excluded_titles = {str(value) for value in discovery.get("exclude_titles", [])}

    priority_pages = [
        str(value).strip()
        for value in discovery.get("priority_pages", []) or []
        if str(value).strip()
    ]
    priority_by_title = {
        title: index
        for index, title in enumerate(priority_pages)
    }
    reference_pages = {
        str(value).strip()
        for value in discovery.get("reference_pages", []) or []
        if str(value).strip()
    }

    def retrieval_role(title: str, source_kind: str) -> str:
        if title in reference_pages:
            return "reference"
        if title.startswith("Guide:") or "guide_category" in source_kind:
            return "guide"
        return "mechanics"

    discovered: dict[str, WikiPageDescriptor] = {}
    for category in discovery.get("guide_categories", []) or []:
        for title in client.category_members(str(category)):
            if include_prefixes and not title.startswith(include_prefixes):
                continue
            if exclude_prefixes and title.startswith(exclude_prefixes):
                continue
            if title in excluded_titles:
                continue
            discovered.setdefault(
                title,
                WikiPageDescriptor(
                    title=title,
                    source_kind="guide_category",
                    retrieval_role=retrieval_role(title, "guide_category"),
                    discovery_priority=priority_by_title.get(title, 10_000),
                ),
            )

    quality_membership: dict[str, list[str]] = defaultdict(list)
    for status, category in (discovery.get("quality_categories") or {}).items():
        for title in client.category_members(str(category)):
            quality_membership[title].append(str(status))

    for title, statuses in quality_membership.items():
        descriptor = discovered.get(title)
        if descriptor is None:
            continue
        selected = max(statuses, key=_quality_priority, default="unknown")
        descriptor.quality_status = selected
        descriptor.quality_flags = sorted(set(statuses))

    for title in discovery.get("explicit_pages", []) or []:
        title = str(title).strip()
        if not title or title in excluded_titles:
            continue
        descriptor = discovered.setdefault(
            title,
            WikiPageDescriptor(
                title=title,
                source_kind="explicit_page",
                retrieval_role=retrieval_role(title, "explicit_page"),
                discovery_priority=priority_by_title.get(title, 10_000),
            ),
        )
        if descriptor.source_kind != "explicit_page":
            descriptor.source_kind = "guide_category+explicit_page"
        descriptor.retrieval_role = retrieval_role(title, descriptor.source_kind)
        descriptor.discovery_priority = priority_by_title.get(title, 10_000)

    role_priority = {"guide": 0, "mechanics": 1, "reference": 2}
    pages = sorted(
        discovered.values(),
        key=lambda row: (
            int(row.discovery_priority),
            role_priority.get(row.retrieval_role, 9),
            row.title.casefold(),
        ),
    )
    if max_pages is not None:
        if int(max_pages) < 1:
            raise ValueError("max_pages must be at least 1 when supplied.")
        pages = pages[: int(max_pages)]
    return pages


def _raw_record_hash(record: dict[str, Any]) -> str:
    payload = "\n".join(
        [
            str(record.get("title", "")),
            str(record.get("revision_id", "")),
            str(record.get("html", "")),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def import_wiki_pages(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
    refresh: bool = False,
    max_pages: int | None = None,
    client: MediaWikiClient | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Discover, incrementally fetch, and persist raw guide pages."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    report_path = Path(report_path).expanduser().resolve()
    manifest = load_source_manifest(manifest_path)

    existing_records = read_jsonl(output_path) if output_path.exists() else []
    existing_by_requested = {
        str(record.get("requested_title") or record.get("title")): record
        for record in existing_records
        if record.get("requested_title") or record.get("title")
    }

    owns_client = client is None
    active_client = client or create_client(manifest)
    errors: list[dict[str, Any]] = []
    try:
        descriptors = discover_pages(active_client, manifest, max_pages=max_pages)
        titles = [descriptor.title for descriptor in descriptors]
        metadata = active_client.page_revision_metadata(titles)

        working_by_requested = dict(existing_by_requested)
        fetched_count = 0
        reused_count = 0
        missing_count = 0
        checkpoint_every = max(
            1,
            int((manifest["wiki"].get("request") or {}).get("checkpoint_every", 5)),
        )

        def checkpoint() -> None:
            snapshot = sorted(
                working_by_requested.values(),
                key=lambda row: str(row.get("title", "")).casefold(),
            )
            write_jsonl(output_path, snapshot)

        for index, descriptor in enumerate(descriptors, start=1):
            current = metadata.get(descriptor.title) or {}
            existing = existing_by_requested.get(descriptor.title)
            revision_id = current.get("revision_id")
            can_reuse = (
                not refresh
                and existing is not None
                and revision_id is not None
                and existing.get("revision_id") == revision_id
            )
            if can_reuse:
                record = dict(existing)
                record["quality_status"] = descriptor.quality_status
                record["quality_flags"] = descriptor.quality_flags
                record["source_kind"] = descriptor.source_kind
                record["retrieval_role"] = descriptor.retrieval_role
                record["discovery_priority"] = descriptor.discovery_priority
                working_by_requested[descriptor.title] = record
                reused_count += 1
                if index % checkpoint_every == 0:
                    checkpoint()
                continue

            if current.get("missing"):
                missing_count += 1
                working_by_requested.pop(descriptor.title, None)
                errors.append(
                    {
                        "title": descriptor.title,
                        "error_type": "missing_page",
                        "message": "MediaWiki metadata reports the page as missing.",
                    }
                )
                continue

            if verbose:
                print(f"[{index}/{len(descriptors)}] Fetching {descriptor.title}")
            try:
                parsed = active_client.parse_page(descriptor.title)
            except MediaWikiAPIError as error:
                errors.append(
                    {
                        "title": descriptor.title,
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
                continue

            fetched_at = utc_now()
            record = {
                "schema_version": 1,
                **parsed,
                "revision_timestamp": current.get("revision_timestamp"),
                "source_kind": descriptor.source_kind,
                "retrieval_role": descriptor.retrieval_role,
                "discovery_priority": descriptor.discovery_priority,
                "quality_status": descriptor.quality_status,
                "quality_flags": descriptor.quality_flags,
                "language": manifest["wiki"]["language"],
                "source_name": manifest["wiki"].get("name", "Official Terraria Wiki"),
                "license": manifest["wiki"].get("license") or {},
                "fetched_at": fetched_at,
            }
            record["content_sha256"] = _raw_record_hash(record)
            working_by_requested[descriptor.title] = record
            fetched_count += 1
            if index % checkpoint_every == 0:
                checkpoint()

        records = [
            working_by_requested[descriptor.title]
            for descriptor in descriptors
            if descriptor.title in working_by_requested
        ]
        records.sort(key=lambda row: str(row.get("title", "")).casefold())
        write_jsonl(output_path, records)

        quality_counts = Counter(str(row.get("quality_status", "unknown")) for row in records)
        source_kind_counts = Counter(str(row.get("source_kind", "unknown")) for row in records)
        retrieval_role_counts = Counter(
            str(row.get("retrieval_role", "unknown")) for row in records
        )
        report = {
            "status": "passed" if not errors else "partial",
            "manifest_path": portable_path(manifest_path),
            "output_path": portable_path(output_path),
            "discovered_pages": len(descriptors),
            "written_pages": len(records),
            "fetched_pages": fetched_count,
            "reused_pages": reused_count,
            "missing_pages": missing_count,
            "failed_pages": len(errors) - missing_count,
            "refresh": bool(refresh),
            "quality_status_counts": dict(quality_counts.most_common()),
            "source_kind_counts": dict(source_kind_counts.most_common()),
            "retrieval_role_counts": dict(retrieval_role_counts.most_common()),
            "selected_titles": [descriptor.title for descriptor in descriptors],
            "raw_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
            "errors": errors,
            "generated_at": utc_now(),
        }
        write_json(report_path, report)
        return report
    finally:
        if owns_client:
            active_client.close()
