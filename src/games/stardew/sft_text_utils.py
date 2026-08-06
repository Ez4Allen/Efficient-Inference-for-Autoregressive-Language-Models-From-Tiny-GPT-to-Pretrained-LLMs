"""Shared text-normalization helpers for Stardew SFT audit/clean/split tooling.

Kept in one module so the audit script, the cleaning pipeline, and the
splitter apply the exact same normalization -- using different normalization
in different tools would make their duplicate/leakage counts incomparable.
"""

from __future__ import annotations

import re

ALLOWED_ROLES = {"system", "user", "assistant"}

_SEASON_WORDS = ("spring", "summer", "fall", "autumn", "winter")
_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")
_NUMBER_RE = re.compile(r"\b\d+\b")
# Filler words that shift a naive word-prefix bucket key without changing the
# underlying question template (e.g. "what can the X be used for" vs "what
# can a X be used for" are the same template).
_FILLER_WORD_RE = re.compile(r"\b(a|an|the|first)\b")


def normalize_text(text: str) -> str:
    """Casefold, collapse whitespace, and strip punctuation."""
    text = text.casefold()
    text = _WHITESPACE_RE.sub(" ", text)
    text = _PUNCT_RE.sub("", text)
    return text.strip()


def extract_single_turn_qa(messages: object) -> tuple[str, str] | None:
    """Validate message structure and return ``(question, answer)``.

    Returns ``None`` when ``messages`` does not follow the DATA_FORMAT.md
    contract: optional leading system message, then alternating user/
    assistant, ending on assistant, all non-empty string content. On success,
    returns the *last* user message and the *last* assistant message (works
    for both single-turn and multi-turn records).
    """

    if not isinstance(messages, list) or not messages:
        return None

    roles: list[str] = []
    contents: list[str] = []

    for message in messages:
        if not isinstance(message, dict):
            return None

        role = message.get("role")
        content = message.get("content")

        if role not in ALLOWED_ROLES:
            return None

        if not isinstance(content, str) or not content.strip():
            return None

        roles.append(role)
        contents.append(content)

    for index, role in enumerate(roles):
        if role == "system" and index != 0:
            return None

    conversation_roles = roles[1:] if roles[0] == "system" else roles
    conversation_contents = contents[1:] if roles[0] == "system" else contents

    if not conversation_roles or conversation_roles[-1] != "assistant":
        return None

    expected = "user"
    for role in conversation_roles:
        if role != expected:
            return None
        expected = "assistant" if expected == "user" else "user"

    last_user = None
    last_assistant = None
    for role, content in zip(conversation_roles, conversation_contents):
        if role == "user":
            last_user = content
        else:
            last_assistant = content

    if last_user is None or last_assistant is None:
        return None

    return last_user, last_assistant


def mask_template(text: str, entity_phrases: list[str]) -> str:
    """Normalize ``text`` and mask out known entity phrases/numbers/seasons.

    Used to compare the *shape* of two questions/answers while ignoring
    which specific entity, quantity, or season they mention -- this is how
    entity-substitution template duplicates (e.g. "Walnut Table" vs "Walnut
    End Table" asked with the same underlying question shape) are detected.
    """

    normalized = normalize_text(text)

    for phrase in sorted(entity_phrases, key=len, reverse=True):
        masked_phrase = normalize_text(phrase)
        if masked_phrase and masked_phrase in normalized:
            normalized = normalized.replace(masked_phrase, "<entity>")

    normalized = _NUMBER_RE.sub("<num>", normalized)

    for season in _SEASON_WORDS:
        normalized = re.sub(rf"\b{season}\b", "<season>", normalized)

    normalized = _FILLER_WORD_RE.sub("", normalized)
    normalized = _WHITESPACE_RE.sub(" ", normalized).strip()

    return normalized
