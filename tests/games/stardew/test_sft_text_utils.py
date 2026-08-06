from __future__ import annotations

from src.games.stardew.sft_text_utils import (
    extract_single_turn_qa,
    mask_template,
    normalize_text,
)


def test_normalize_text_folds_case_and_punctuation() -> None:
    assert normalize_text("What's up, Farmer?!") == "whats up farmer"


def test_normalize_text_collapses_whitespace() -> None:
    assert normalize_text("a   b\n\tc") == "a b c"


def test_extract_single_turn_qa_happy_path() -> None:
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "question?"},
        {"role": "assistant", "content": "answer."},
    ]
    assert extract_single_turn_qa(messages) == ("question?", "answer.")


def test_extract_single_turn_qa_without_system_message() -> None:
    messages = [
        {"role": "user", "content": "question?"},
        {"role": "assistant", "content": "answer."},
    ]
    assert extract_single_turn_qa(messages) == ("question?", "answer.")


def test_extract_single_turn_qa_multi_turn_returns_last_pair() -> None:
    messages = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    assert extract_single_turn_qa(messages) == ("q2", "a2")


def test_extract_single_turn_qa_rejects_empty_list() -> None:
    assert extract_single_turn_qa([]) is None


def test_extract_single_turn_qa_rejects_wrong_role_order() -> None:
    messages = [
        {"role": "assistant", "content": "answer first?"},
        {"role": "user", "content": "question second"},
    ]
    assert extract_single_turn_qa(messages) is None


def test_extract_single_turn_qa_rejects_non_assistant_ending() -> None:
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "trailing question"},
    ]
    assert extract_single_turn_qa(messages) is None


def test_extract_single_turn_qa_rejects_empty_content() -> None:
    messages = [
        {"role": "user", "content": "  "},
        {"role": "assistant", "content": "a"},
    ]
    assert extract_single_turn_qa(messages) is None


def test_extract_single_turn_qa_rejects_system_not_first() -> None:
    messages = [
        {"role": "user", "content": "q"},
        {"role": "system", "content": "sys"},
        {"role": "assistant", "content": "a"},
    ]
    assert extract_single_turn_qa(messages) is None


def test_extract_single_turn_qa_rejects_invalid_role() -> None:
    messages = [
        {"role": "user", "content": "q"},
        {"role": "tool", "content": "a"},
    ]
    assert extract_single_turn_qa(messages) is None


def test_mask_template_masks_entity_number_and_season() -> None:
    result = mask_template(
        "How many Slime Incubators can I place in Spring 5?",
        entity_phrases=["Slime Incubator"],
    )
    assert "<entity>" in result
    assert "<num>" in result
    assert "<season>" in result
    assert "slime incubator" not in result


def test_mask_template_strips_filler_words_for_bucketing() -> None:
    left = mask_template("What can the Wool be used for?", entity_phrases=["Wool"])
    right = mask_template("What can a Wool be used for?", entity_phrases=["Wool"])
    assert left == right
