from __future__ import annotations

from pathlib import Path

import pytest

from src.training.tiny_qwen_draft import (
    load_tiny_qwen_draft_training_config,
)


def test_tiny_qwen_draft_training_config_loads(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    train.write_text("{}\n", encoding="utf-8")
    config_path = tmp_path / "draft.yaml"
    config_path.write_text(
        f"""
model:
  target_model_name_or_path: Qwen/Qwen3-4B
  tokenizer_name_or_path: Qwen/Qwen3-4B
  hidden_size: 16
  intermediate_size: 32
  num_hidden_layers: 2
  num_attention_heads: 4
  num_key_value_heads: 2
  max_position_embeddings: 64
data:
  train_path: {train}
  max_length: 32
training:
  output_dir: {tmp_path / 'output'}
  max_steps: 5
  gradient_accumulation_steps: 2
""",
        encoding="utf-8",
    )

    config = load_tiny_qwen_draft_training_config(config_path)

    assert config.hidden_size == 16
    assert config.train_path == train.resolve()
    assert config.max_steps == 5
    assert config.validation_path is None


def test_tiny_qwen_draft_training_config_rejects_long_examples(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "draft.yaml"
    config_path.write_text(
        f"""
model:
  target_model_name_or_path: target
  tokenizer_name_or_path: tokenizer
  max_position_embeddings: 16
data:
  train_path: {tmp_path / 'train.jsonl'}
  max_length: 32
training:
  output_dir: {tmp_path / 'output'}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="max_length"):
        load_tiny_qwen_draft_training_config(config_path)


def test_distillation_split_validation_rejects_eval_records(tmp_path: Path) -> None:
    from src.training.tiny_qwen_draft import _validate_distillation_split

    source = tmp_path / "teacher.jsonl"
    source.write_text(
        '{"id":"eval-1","split":"eval","messages":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 'train'"):
        _validate_distillation_split(source, expected_split="train")


def test_distillation_split_validation_and_file_hash(tmp_path: Path) -> None:
    from src.training.tiny_qwen_draft import (
        _file_sha256,
        _validate_distillation_split,
    )

    source = tmp_path / "teacher_train.jsonl"
    source.write_text(
        '{"id":"train-1","split":"train","messages":[]}\n',
        encoding="utf-8",
    )

    records = _validate_distillation_split(source, expected_split="train")
    assert records[0]["id"] == "train-1"
    assert len(_file_sha256(source)) == 64
