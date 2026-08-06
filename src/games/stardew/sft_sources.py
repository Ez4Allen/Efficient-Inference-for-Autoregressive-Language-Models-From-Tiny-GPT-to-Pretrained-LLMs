"""Deterministic canonicalization of Stardew Valley Wiki source URLs.

The SFT candidate pool cites the Official Stardew Valley Wiki
(``https://stardewvalleywiki.com/<Page_Title>``). The same page has been
cited with inconsistent capitalization (``Slime_Incubator`` vs
``Slime_incubator``) and inconsistent space/underscore encoding. Treating
these as distinct sources breaks source-overlap accounting and lets a
group-aware splitter place two records from the same page in different
splits.

This module never invents redirect targets, section titles, revision IDs,
or game versions. It only normalizes the literal URL text.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import unquote, urlparse

ALLOWED_HOSTS = {"stardewvalleywiki.com", "www.stardewvalleywiki.com"}


@dataclass(frozen=True)
class CanonicalSource(object):
    ok: bool
    original_url: str
    # Case-insensitive grouping key. Stable identity for "is this the same
    # page as that other citation", independent of casing.
    group_key: str | None
    # Human-readable canonical title, e.g. "Slime_Incubator". Only the first
    # character's case is normalized (MediaWiki auto-capitalizes the first
    # character of a title); remaining characters keep their observed case
    # because MediaWiki titles are case-sensitive beyond that point and we
    # must not invent a redirect target.
    display_title: str | None
    issue: str | None


def canonicalize_source_url(url: str) -> CanonicalSource:
    if not isinstance(url, str) or not url.strip():
        return CanonicalSource(
            ok=False, original_url=url, group_key=None,
            display_title=None, issue="empty_or_non_string_url",
        )

    raw = url.strip()
    parsed = urlparse(raw)

    if parsed.scheme not in {"http", "https"}:
        return CanonicalSource(
            ok=False, original_url=raw, group_key=None,
            display_title=None, issue="unsupported_scheme",
        )

    if parsed.netloc.lower() not in ALLOWED_HOSTS:
        return CanonicalSource(
            ok=False, original_url=raw, group_key=None,
            display_title=None, issue="unsupported_host",
        )

    path = parsed.path.lstrip("/")

    if not path:
        return CanonicalSource(
            ok=False, original_url=raw, group_key=None,
            display_title=None, issue="missing_page_title",
        )

    # Fragments (#section) and query parameters (?action=edit, ?oldid=..)
    # do not change which page the citation refers to; drop both.
    title = unquote(path)
    title = title.replace(" ", "_")
    while "__" in title:
        title = title.replace("__", "_")

    group_key = title.casefold()
    display_title = title[:1].upper() + title[1:] if title else title

    return CanonicalSource(
        ok=True, original_url=raw, group_key=group_key,
        display_title=display_title, issue=None,
    )


def build_canonical_title_index(all_urls: list[str]) -> dict[str, str]:
    """Resolve one canonical display title per ``group_key`` across a corpus.

    A single record only ever cites one literal casing of a page. To decide
    which casing is "the" canonical one for a page, this must look at every
    URL cited anywhere in the corpus, not just one record's citations.
    Ties are broken by taking the lexicographically smallest observed
    display title, which is deterministic and reproducible but does not
    claim to know the real MediaWiki redirect target.
    """

    variants_by_key: dict[str, set[str]] = {}

    for url in all_urls:
        result = canonicalize_source_url(url)

        if not result.ok:
            continue

        assert result.group_key is not None
        assert result.display_title is not None
        variants_by_key.setdefault(result.group_key, set()).add(result.display_title)

    return {
        key: min(variants)
        for key, variants in variants_by_key.items()
    }


def canonicalize_record_sources(
    urls: list[str],
    canonical_index: dict[str, str],
) -> tuple[list[str], list[str], list[str]]:
    """Canonicalize one record's ``source_urls`` using a prebuilt corpus index.

    Returns ``(source_pages, group_keys, issues)`` where ``source_pages`` is
    a deduplicated, sorted list of canonical display titles, ``group_keys``
    is the matching sorted list of case-insensitive grouping keys, and
    ``issues`` lists any per-URL problems (e.g. ``"unsupported_host"``).
    """

    keys: set[str] = set()
    issues: list[str] = []

    for url in urls:
        result = canonicalize_source_url(url)

        if not result.ok:
            issues.append(result.issue or "invalid_source_url")
            continue

        assert result.group_key is not None
        keys.add(result.group_key)

    group_keys = sorted(keys)
    source_pages = [canonical_index.get(key, key) for key in group_keys]

    return source_pages, group_keys, issues
