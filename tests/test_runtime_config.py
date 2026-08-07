from __future__ import annotations

from pathlib import Path

import pytest

from src.models.runtime_config import load_qwen_pair_config


def test_pair_config_loads_and_expands_environment(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("DRAFT_MODEL", "Qwen/Qwen3-0.6B")
    path = tmp_path / "pair.yaml"
    path.write_text(
        """
models:
  draft:
    model_name_or_path: ${DRAFT_MODEL}
    tokenizer_name_or_path: Qwen/Qwen3-4B
    attn_implementation: sdpa
  target:
    model_name_or_path: Qwen/Qwen3-4B
    attn_implementation: eager
runtime:
  device: cpu
  dtype: fp32
generation:
  engine: speculative
  max_new_tokens: 20
  draft_tokens_per_round: 3
  verification_mode: block
""",
        encoding="utf-8",
    )

    config = load_qwen_pair_config(path)

    assert config.draft.model_name_or_path == "Qwen/Qwen3-0.6B"
    assert config.draft.tokenizer_name_or_path == "Qwen/Qwen3-4B"
    assert config.target.model_name_or_path == "Qwen/Qwen3-4B"
    assert config.draft.attn_implementation == "sdpa"
    assert config.target.attn_implementation == "eager"
    assert config.generation.engine == "speculative"
    assert config.generation.draft_tokens_per_round == 3
    assert config.generation.verification_mode == "block"


def test_pair_config_rejects_unknown_engine(tmp_path: Path) -> None:
    path = tmp_path / "pair.yaml"
    path.write_text(
        """
models:
  draft: {model_name_or_path: draft}
  target: {model_name_or_path: target}
generation: {engine: magic}
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="generation.engine"):
        load_qwen_pair_config(path)


def test_pair_config_defaults_to_exact_verification(tmp_path: Path) -> None:
    path = tmp_path / "pair.yaml"
    path.write_text(
        """
models:
  draft: {model_name_or_path: draft}
  target: {model_name_or_path: target}
""",
        encoding="utf-8",
    )

    config = load_qwen_pair_config(path)

    assert config.generation.verification_mode == "exact"


def test_pair_config_rejects_unknown_verification_mode(tmp_path: Path) -> None:
    path = tmp_path / "pair.yaml"
    path.write_text(
        """
models:
  draft: {model_name_or_path: draft}
  target: {model_name_or_path: target}
generation: {verification_mode: magic}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="verification_mode"):
        load_qwen_pair_config(path)
