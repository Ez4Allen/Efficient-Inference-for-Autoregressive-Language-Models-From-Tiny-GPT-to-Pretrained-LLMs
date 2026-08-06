from __future__ import annotations

from src.games.stardew.sft_sources import (
    build_canonical_title_index,
    canonicalize_record_sources,
    canonicalize_source_url,
)


def test_valid_wiki_url_is_canonicalized() -> None:
    result = canonicalize_source_url("https://stardewvalleywiki.com/Slime_Incubator")
    assert result.ok
    assert result.group_key == "slime_incubator"
    assert result.display_title == "Slime_Incubator"


def test_fragment_and_query_are_dropped() -> None:
    result = canonicalize_source_url(
        "https://stardewvalleywiki.com/Slime_Incubator?action=edit#History"
    )
    assert result.ok
    assert result.group_key == "slime_incubator"


def test_case_variants_share_the_same_group_key() -> None:
    a = canonicalize_source_url("https://stardewvalleywiki.com/Slime_Incubator")
    b = canonicalize_source_url("https://stardewvalleywiki.com/Slime_incubator")
    assert a.group_key == b.group_key
    assert a.display_title != b.display_title


def test_spaces_and_underscores_are_normalized_the_same_way() -> None:
    a = canonicalize_source_url("https://stardewvalleywiki.com/Spring%20Onion")
    b = canonicalize_source_url("https://stardewvalleywiki.com/Spring_Onion")
    assert a.group_key == b.group_key == "spring_onion"


def test_non_wiki_host_is_rejected() -> None:
    result = canonicalize_source_url("https://example.com/Slime_Incubator")
    assert not result.ok
    assert result.issue == "unsupported_host"


def test_non_http_scheme_is_rejected() -> None:
    result = canonicalize_source_url("ftp://stardewvalleywiki.com/Slime_Incubator")
    assert not result.ok
    assert result.issue == "unsupported_scheme"


def test_url_without_page_title_is_rejected() -> None:
    result = canonicalize_source_url("https://stardewvalleywiki.com/")
    assert not result.ok
    assert result.issue == "missing_page_title"


def test_empty_url_is_rejected() -> None:
    result = canonicalize_source_url("")
    assert not result.ok
    assert result.issue == "empty_or_non_string_url"


def test_canonical_title_index_picks_smallest_variant_deterministically() -> None:
    urls = [
        "https://stardewvalleywiki.com/Statue_Of_Perfection",
        "https://stardewvalleywiki.com/Statue_of_perfection",
        "https://stardewvalleywiki.com/Statue_of_Perfection",
    ]
    index = build_canonical_title_index(urls)
    key = "statue_of_perfection"
    assert index[key] == min(
        ["Statue_Of_Perfection", "Statue_of_perfection", "Statue_of_Perfection"]
    )

    # Rebuilding from the same URLs must yield the same answer.
    index_again = build_canonical_title_index(list(reversed(urls)))
    assert index_again[key] == index[key]


def test_canonicalize_record_sources_uses_corpus_wide_canonical_title() -> None:
    index = {"slime_incubator": "Slime_Incubator"}

    pages, keys, issues = canonicalize_record_sources(
        ["https://stardewvalleywiki.com/Slime_incubator"], index
    )

    assert pages == ["Slime_Incubator"]
    assert keys == ["slime_incubator"]
    assert issues == []


def test_canonicalize_record_sources_reports_issues_for_bad_urls() -> None:
    pages, keys, issues = canonicalize_record_sources(
        ["not a url", "https://example.com/Foo"], {}
    )
    assert pages == []
    assert keys == []
    assert len(issues) == 2
