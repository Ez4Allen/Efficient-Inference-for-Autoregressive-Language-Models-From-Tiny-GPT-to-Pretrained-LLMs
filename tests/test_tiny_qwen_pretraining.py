from __future__ import annotations

from pathlib import Path

import pytest

from src.training.tiny_qwen_pretraining import load_tiny_qwen_pretraining_config


def test_pretraining_config_loads(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"id":"a","split":"train","text":"abc"}\n'
        '{"id":"b","split":"validation","text":"def"}\n',
        encoding="utf-8",
    )
    path = tmp_path / "pretrain.yaml"
    path.write_text(
        f"""
model:
  teacher_model_name_or_path: Qwen/Qwen3-0.6B
  tokenizer_name_or_path: Qwen/Qwen3-0.6B
  hidden_size: 16
  intermediate_size: 32
  num_hidden_layers: 2
  num_attention_heads: 4
  num_key_value_heads: 2
  max_position_embeddings: 64
data:
  corpus_path: {corpus}
  max_length: 32
  stride: 16
training:
  output_dir: {tmp_path / 'output'}
  max_steps: 5
""",
        encoding="utf-8",
    )
    config = load_tiny_qwen_pretraining_config(path)
    assert config.max_length == 32
    assert config.stride == 16
    assert config.max_steps == 5


def test_pretraining_config_rejects_stride_larger_than_length(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        f"""
model:
  teacher_model_name_or_path: teacher
  tokenizer_name_or_path: tokenizer
data:
  corpus_path: {tmp_path / 'corpus.jsonl'}
  max_length: 16
  stride: 32
training:
  output_dir: {tmp_path / 'output'}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="stride"):
        load_tiny_qwen_pretraining_config(path)
