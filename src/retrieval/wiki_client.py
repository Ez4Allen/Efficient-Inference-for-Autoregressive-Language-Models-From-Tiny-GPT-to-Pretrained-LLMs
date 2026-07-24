"""Rate-limited MediaWiki API client used by the guide importer."""

from __future__ import annotations

import time
from collections.abc import Iterable
from typing import Any
from urllib.parse import quote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class MediaWikiAPIError(RuntimeError):
    """Raised when a MediaWiki API response is malformed or reports an error."""


class MediaWikiClient:
    """Small, polite client for read-only MediaWiki API operations."""

    def __init__(
        self,
        *,
        api_url: str,
        article_base_url: str,
        user_agent: str,
        timeout_seconds: float = 30.0,
        request_delay_seconds: float = 0.35,
        max_retries: int = 4,
        session: requests.Session | None = None,
    ) -> None:
        if not str(user_agent).strip():
            raise ValueError("A descriptive user_agent is required.")

        self.api_url = str(api_url).strip()
        self.article_base_url = str(article_base_url).rstrip("/") + "/"
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.request_delay_seconds = max(0.0, float(request_delay_seconds))
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "User-Agent": str(user_agent).strip(),
                "Accept": "application/json",
            }
        )

        retry = Retry(
            total=max(0, int(max_retries)),
            connect=max(0, int(max_retries)),
            read=max(0, int(max_retries)),
            status=max(0, int(max_retries)),
            backoff_factor=0.8,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.mount("http://", HTTPAdapter(max_retries=retry))
        self._last_request_finished = 0.0

    def close(self) -> None:
        self.session.close()

    def __enter__(self) -> "MediaWikiClient":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_finished
        delay = self.request_delay_seconds - elapsed
        if delay > 0:
            time.sleep(delay)

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        query = {
            "format": "json",
            "formatversion": 2,
            "utf8": 1,
            **params,
        }
        self._throttle()
        try:
            response = self.session.get(
                self.api_url,
                params=query,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as error:
            raise MediaWikiAPIError(
                f"MediaWiki request failed for action={params.get('action')!r}: {error}"
            ) from error
        except ValueError as error:
            raise MediaWikiAPIError("MediaWiki returned non-JSON content.") from error
        finally:
            self._last_request_finished = time.monotonic()

        if not isinstance(payload, dict):
            raise MediaWikiAPIError("MediaWiki response must be a JSON object.")
        if "error" in payload:
            raise MediaWikiAPIError(f"MediaWiki API error: {payload['error']}")
        return payload

    def article_url(self, title: str) -> str:
        slug = str(title).replace(" ", "_")
        return self.article_base_url + quote(slug, safe=":/()'!,-")

    def category_members(self, category: str) -> list[str]:
        """Return all page titles directly contained in *category*."""

        titles: list[str] = []
        continuation: dict[str, Any] = {}
        while True:
            payload = self._request(
                {
                    "action": "query",
                    "list": "categorymembers",
                    "cmtitle": category,
                    "cmtype": "page",
                    "cmlimit": "max",
                    **continuation,
                }
            )
            members = payload.get("query", {}).get("categorymembers", [])
            for member in members:
                title = member.get("title") if isinstance(member, dict) else None
                if title:
                    titles.append(str(title))

            next_values = payload.get("continue")
            if not isinstance(next_values, dict):
                break
            continuation = next_values

        return list(dict.fromkeys(titles))

    @staticmethod
    def _batches(values: list[str], size: int = 50) -> Iterable[list[str]]:
        for start in range(0, len(values), size):
            yield values[start : start + size]

    def page_revision_metadata(self, titles: list[str]) -> dict[str, dict[str, Any]]:
        """Return current page/revision metadata keyed by requested title.

        MediaWiki redirects and title normalization are followed. Missing pages
        are represented with ``missing=True``.
        """

        output: dict[str, dict[str, Any]] = {}
        for batch in self._batches(list(dict.fromkeys(titles))):
            payload = self._request(
                {
                    "action": "query",
                    "prop": "info|revisions",
                    "inprop": "url",
                    "rvprop": "ids|timestamp",
                    "redirects": 1,
                    "titles": "|".join(batch),
                }
            )
            query = payload.get("query", {})
            alias_map = {title: title for title in batch}
            for key in ("normalized", "redirects"):
                for mapping in query.get(key, []) or []:
                    if isinstance(mapping, dict) and mapping.get("from") and mapping.get("to"):
                        alias_map[str(mapping["from"])] = str(mapping["to"])

            def resolve(title: str) -> str:
                seen: set[str] = set()
                current = title
                while current in alias_map and alias_map[current] != current and current not in seen:
                    seen.add(current)
                    current = alias_map[current]
                return current

            pages = {
                str(page.get("title")): page
                for page in query.get("pages", []) or []
                if isinstance(page, dict) and page.get("title")
            }
            for requested in batch:
                canonical = resolve(requested)
                page = pages.get(canonical) or pages.get(requested)
                if not page:
                    output[requested] = {
                        "requested_title": requested,
                        "title": canonical,
                        "missing": True,
                    }
                    continue
                revisions = page.get("revisions") or []
                revision = revisions[0] if revisions else {}
                output[requested] = {
                    "requested_title": requested,
                    "title": str(page.get("title", canonical)),
                    "page_id": page.get("pageid"),
                    "revision_id": revision.get("revid"),
                    "revision_timestamp": revision.get("timestamp"),
                    "source_url": page.get("fullurl") or self.article_url(canonical),
                    "missing": bool(page.get("missing", False)),
                }
        return output

    def parse_page(self, title: str) -> dict[str, Any]:
        """Fetch rendered article HTML and parse metadata for *title*."""

        payload = self._request(
            {
                "action": "parse",
                "page": title,
                "prop": "text|sections|categories|displaytitle|revid|properties",
                "redirects": 1,
                "disableeditsection": 1,
            }
        )
        parsed = payload.get("parse")
        if not isinstance(parsed, dict):
            raise MediaWikiAPIError(f"No parse payload returned for {title!r}.")

        html = parsed.get("text")
        if isinstance(html, dict):
            html = html.get("*")
        if not isinstance(html, str):
            raise MediaWikiAPIError(f"No rendered HTML returned for {title!r}.")

        categories: list[str] = []
        for category in parsed.get("categories", []) or []:
            if not isinstance(category, dict):
                continue
            category_name = category.get("category") or category.get("*")
            if category_name:
                categories.append(str(category_name).replace("_", " "))

        canonical_title = str(parsed.get("title") or title)
        return {
            "requested_title": title,
            "title": canonical_title,
            "page_id": parsed.get("pageid"),
            "revision_id": parsed.get("revid"),
            "display_title": parsed.get("displaytitle"),
            "source_url": self.article_url(canonical_title),
            "html": html,
            "sections_api": parsed.get("sections") or [],
            "categories": categories,
            "properties": parsed.get("properties") or [],
        }
