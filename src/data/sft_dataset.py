
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase
else:
    PreTrainedTokenizerBase = Any


IGNORE_INDEX = -100
ALLOWED_ROLES = {"system", "user", "assistant"}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}, line {line_number}"
                ) from error

            if not isinstance(record, dict):
                raise TypeError(
                    f"{path}, line {line_number}: "
                    "each JSONL record must be an object"
                )

            records.append(record)

    if not records:
        raise ValueError(f"Dataset is empty: {path}")

    return records


def validate_messages(
    messages: Any,
    record_id: str,
) -> list[dict[str, str]]:
    """
    Validate a chat conversation.

    Supported pattern:

    system (optional)
    user
    assistant
    user
    assistant
    ...
    """

    if not isinstance(messages, list):
        raise TypeError(
            f"{record_id}: 'messages' must be a list"
        )

    if not messages:
        raise ValueError(
            f"{record_id}: 'messages' cannot be empty"
        )

    validated: list[dict[str, str]] = []

    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise TypeError(
                f"{record_id}: message {index} must be a dictionary"
            )

        role = message.get("role")
        content = message.get("content")

        if role not in ALLOWED_ROLES:
            raise ValueError(
                f"{record_id}: invalid role at message "
                f"{index}: {role!r}"
            )

        if not isinstance(content, str):
            raise TypeError(
                f"{record_id}: message {index} content "
                "must be a string"
            )

        content = content.strip()

        if not content:
            raise ValueError(
                f"{record_id}: message {index} content is empty"
            )

        validated.append(
            {
                "role": role,
                "content": content,
            }
        )

    # System message is optional, but it may only appear first.
    for index, message in enumerate(validated):
        if message["role"] == "system" and index != 0:
            raise ValueError(
                f"{record_id}: system message may only appear first"
            )

    conversation = validated

    if conversation[0]["role"] == "system":
        conversation_without_system = conversation[1:]
    else:
        conversation_without_system = conversation

    if not conversation_without_system:
        raise ValueError(
            f"{record_id}: conversation contains only a system message"
        )

    expected_role = "user"

    for index, message in enumerate(conversation_without_system):
        if message["role"] != expected_role:
            raise ValueError(
                f"{record_id}: expected role {expected_role!r}, "
                f"but found {message['role']!r} at conversation "
                f"message {index}"
            )

        expected_role = (
            "assistant"
            if expected_role == "user"
            else "user"
        )

    if conversation_without_system[-1]["role"] != "assistant":
        raise ValueError(
            f"{record_id}: final message must be from assistant"
        )

    return validated


def _extract_input_ids(tokenized: Any) -> list[int]:
    """
    Convert tokenizer output into a flat list[int].

    Supported tokenizer outputs include:

    - list[int]
    - list[list[int]]
    - tuple[int]
    - torch.Tensor
    - BatchEncoding
    - dictionaries containing input_ids
    """

    if isinstance(tokenized, Mapping):
        if "input_ids" not in tokenized:
            raise TypeError(
                "Tokenizer output mapping does not contain "
                "'input_ids'"
            )

        tokenized = tokenized["input_ids"]

    elif hasattr(tokenized, "input_ids"):
        tokenized = tokenized.input_ids

    if isinstance(tokenized, torch.Tensor):
        tokenized = (
            tokenized
            .detach()
            .cpu()
            .tolist()
        )

    elif hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()

    if isinstance(tokenized, tuple):
        tokenized = list(tokenized)

    # Remove a single batch dimension:
    # [[1, 2, 3]] -> [1, 2, 3]
    if (
        isinstance(tokenized, list)
        and len(tokenized) == 1
        and isinstance(tokenized[0], (list, tuple))
    ):
        tokenized = list(tokenized[0])

    if not isinstance(tokenized, list):
        raise TypeError(
            "Unsupported tokenizer output type: "
            f"{type(tokenized).__name__}"
        )

    if not tokenized:
        raise ValueError(
            "Tokenizer returned an empty token sequence"
        )

    result: list[int] = []

    for index, token_id in enumerate(tokenized):
        if isinstance(token_id, bool):
            raise TypeError(
                f"Token ID at index {index} is boolean"
            )

        if not isinstance(token_id, int):
            raise TypeError(
                "Tokenizer output is not a flat integer list. "
                f"Item {index} has type "
                f"{type(token_id).__name__}: {token_id!r}"
            )

        result.append(token_id)

    return result


def _apply_chat_template_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
) -> list[int]:
    """
    Apply a model-specific chat template and return flat token IDs.

    The fallback supports Transformers versions where
    return_dict is unavailable for apply_chat_template.
    """

    common_kwargs = {
        "tokenize": True,
        "add_generation_prompt": add_generation_prompt,
    }

    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            return_dict=True,
            **common_kwargs,
        )

    except TypeError:
        tokenized = tokenizer.apply_chat_template(
            messages,
            **common_kwargs,
        )

    return _extract_input_ids(tokenized)


def build_sft_example(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    record_id: str,
    truncation_mode: str = "right",
) -> dict[str, torch.Tensor]:
    """
    Convert one chat conversation into model training tensors.

    System and user tokens:
        labels = -100

    Assistant response tokens:
        labels = corresponding input token IDs
    """

    if max_length <= 0:
        raise ValueError(
            "max_length must be greater than zero"
        )

    truncation_mode = str(truncation_mode).strip().casefold()
    if truncation_mode not in {"right", "preserve_assistant"}:
        raise ValueError(
            "truncation_mode must be 'right' or 'preserve_assistant'."
        )

    full_ids = _apply_chat_template_ids(
        tokenizer=tokenizer,
        messages=messages,
        add_generation_prompt=False,
    )

    labels = [IGNORE_INDEX] * len(full_ids)

    for message_index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        messages_before_assistant = messages[:message_index]
        messages_through_assistant = messages[
            : message_index + 1
        ]

        # Includes the assistant role/header, but not its answer.
        assistant_prefix_ids = _apply_chat_template_ids(
            tokenizer=tokenizer,
            messages=messages_before_assistant,
            add_generation_prompt=True,
        )

        # Includes the assistant role/header and answer.
        assistant_complete_ids = _apply_chat_template_ids(
            tokenizer=tokenizer,
            messages=messages_through_assistant,
            add_generation_prompt=False,
        )

        assistant_start = len(assistant_prefix_ids)
        assistant_end = len(assistant_complete_ids)

        if assistant_start > assistant_end:
            raise ValueError(
                f"{record_id}: assistant prefix is longer than "
                f"the completed assistant sequence"
            )

        if (
            assistant_complete_ids[:assistant_start]
            != assistant_prefix_ids
        ):
            raise ValueError(
                f"{record_id}: chat template is not prefix-stable "
                f"for assistant message at index {message_index}"
            )

        if full_ids[:assistant_end] != assistant_complete_ids:
            raise ValueError(
                f"{record_id}: full conversation tokenization "
                f"does not match the assistant sequence at "
                f"message index {message_index}"
            )

        if assistant_end == assistant_start:
            raise ValueError(
                f"{record_id}: assistant response produced "
                "no supervised tokens"
            )

        labels[assistant_start:assistant_end] = full_ids[
            assistant_start:assistant_end
        ]

    if len(full_ids) <= max_length or truncation_mode == "right":
        input_ids = full_ids[:max_length]
        labels = labels[:max_length]
    else:
        # Sequence-level distillation supervises only assistant tokens.  When a
        # long evidence prompt exceeds the context budget, preserve the answer
        # and the most recent prompt suffix instead of silently truncating every
        # supervised token.
        input_ids = full_ids[-max_length:]
        labels = labels[-max_length:]
    attention_mask = [1] * len(input_ids)

    supervised_token_count = sum(
        label != IGNORE_INDEX
        for label in labels
    )

    if supervised_token_count == 0:
        raise ValueError(
            f"{record_id}: no assistant tokens remain after "
            f"truncation to max_length={max_length}"
        )

    return {
        "input_ids": torch.tensor(
            input_ids,
            dtype=torch.long,
        ),
        "attention_mask": torch.tensor(
            attention_mask,
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            labels,
            dtype=torch.long,
        ),
    }


class SFTJsonlDataset(Dataset):
    """
    Generic JSONL dataset for supervised chat fine-tuning.

    The dataset is domain-independent. Any game or other domain can
    use it as long as every record contains a valid messages field.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 512,
        truncation_mode: str = "right",
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_mode = str(truncation_mode).strip().casefold()

        records = load_jsonl(self.path)

        self.examples: list[
            dict[str, torch.Tensor]
        ] = []

        for index, record in enumerate(records):
            record_id = str(
                record.get(
                    "id",
                    f"{self.path.name}:{index + 1}",
                )
            )

            messages = validate_messages(
                messages=record.get("messages"),
                record_id=record_id,
            )

            example = build_sft_example(
                messages=messages,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                record_id=record_id,
                truncation_mode=self.truncation_mode,
            )

            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        return self.examples[index]


@dataclass
class SFTDataCollator:
    """
    Dynamically pad SFT examples into one batch.
    """

    tokenizer: PreTrainedTokenizerBase
    label_pad_token_id: int = IGNORE_INDEX

    def __post_init__(self) -> None:
        pad_token_id = self.tokenizer.pad_token_id

        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                "Tokenizer must provide pad_token_id "
                "or eos_token_id"
            )

        self.pad_token_id = int(pad_token_id)

    def __call__(
        self,
        features: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if not features:
            raise ValueError(
                "Cannot create a batch from an empty feature list"
            )

        required_keys = {
            "input_ids",
            "attention_mask",
            "labels",
        }

        for index, feature in enumerate(features):
            missing_keys = required_keys - feature.keys()

            if missing_keys:
                raise KeyError(
                    f"Feature {index} is missing keys: "
                    f"{sorted(missing_keys)}"
                )

        input_ids = pad_sequence(
            [
                feature["input_ids"]
                for feature in features
            ],
            batch_first=True,
            padding_value=self.pad_token_id,
        )

        attention_mask = pad_sequence(
            [
                feature["attention_mask"]
                for feature in features
            ],
            batch_first=True,
            padding_value=0,
        )

        labels = pad_sequence(
            [
                feature["labels"]
                for feature in features
            ],
            batch_first=True,
            padding_value=self.label_pad_token_id,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
