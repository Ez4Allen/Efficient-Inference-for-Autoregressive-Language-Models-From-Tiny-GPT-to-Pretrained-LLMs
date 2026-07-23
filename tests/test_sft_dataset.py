from __future__ import annotations

import pytest
import torch

from src.data.sft_dataset import (
    IGNORE_INDEX,
    SFTDataCollator,
    build_sft_example,
    validate_messages,
)


class FakeChatTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    _role_ids = {"system": 10, "user": 20, "assistant": 30}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        return_dict: bool = False,
    ):
        assert tokenize is True
        ids = [1]
        for message in messages:
            ids.append(self._role_ids[message["role"]])
            ids.extend(100 + ord(character) % 50 for character in message["content"])
            ids.append(2)
        if add_generation_prompt:
            ids.append(self._role_ids["assistant"])
        return {"input_ids": ids} if return_dict else ids


def test_validate_messages_enforces_role_order() -> None:
    messages = validate_messages(
        [
            {"role": "system", "content": "Rules"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ],
        record_id="example",
    )
    assert messages[-1]["role"] == "assistant"

    with pytest.raises(ValueError, match="expected role"):
        validate_messages(
            [
                {"role": "user", "content": "Question"},
                {"role": "user", "content": "Duplicate"},
            ],
            record_id="bad",
        )


def test_build_sft_example_masks_non_assistant_tokens() -> None:
    tokenizer = FakeChatTokenizer()
    messages = [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "OK"},
    ]

    example = build_sft_example(
        messages,
        tokenizer,
        max_length=64,
        record_id="example",
    )

    supervised = example["labels"] != IGNORE_INDEX
    assert supervised.any()
    assert torch.equal(
        example["labels"][supervised],
        example["input_ids"][supervised],
    )
    assert (example["labels"][~supervised] == IGNORE_INDEX).all()


def test_data_collator_pads_ids_masks_and_labels() -> None:
    collator = SFTDataCollator(FakeChatTokenizer())
    batch = collator(
        [
            {
                "input_ids": torch.tensor([1, 2, 3]),
                "attention_mask": torch.tensor([1, 1, 1]),
                "labels": torch.tensor([IGNORE_INDEX, 2, 3]),
            },
            {
                "input_ids": torch.tensor([4, 5]),
                "attention_mask": torch.tensor([1, 1]),
                "labels": torch.tensor([IGNORE_INDEX, 5]),
            },
        ]
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 5, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert batch["labels"].tolist() == [
        [IGNORE_INDEX, 2, 3],
        [IGNORE_INDEX, 5, IGNORE_INDEX],
    ]
