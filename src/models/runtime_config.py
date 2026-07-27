
"""Configuration for the paired Qwen GameGuideLM runtime."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelEndpointConfig:
    model_name_or_path: str
    adapter_path: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    load_in_4bit: bool = False


@dataclass(frozen=True)
class GenerationConfig:
    engine: str = "target"
    max_new_tokens: int = 256
    draft_tokens_per_round: int = 4
    enable_thinking: bool = False


@dataclass(frozen=True)
class GroundingConfig:
    require_citations: bool = True
    fallback_on_error: bool = True
    max_answer_chars: int = 6000
    prompt_mode: str = "evidence_only"
    evidence_policy: str = "compact"
    max_evidence_sources: int = 6
    max_evidence_characters: int = 14_000
    max_repair_attempts: int = 1


@dataclass(frozen=True)
class QwenPairConfig:
    draft: ModelEndpointConfig
    target: ModelEndpointConfig
    device: str = "auto"
    dtype: str = "auto"
    generation: GenerationConfig = GenerationConfig()
    grounding: GroundingConfig = GroundingConfig()


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        expanded = os.path.expandvars(os.path.expanduser(value))
        if "$" in expanded:
            raise ValueError(f"Unresolved environment variable in config: {value}")
        return expanded
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _endpoint(payload: dict[str, Any], name: str) -> ModelEndpointConfig:
    if not isinstance(payload, dict):
        raise TypeError(f"models.{name} must be a mapping.")
    model_reference = str(payload.get("model_name_or_path", "")).strip()
    if not model_reference:
        raise ValueError(f"models.{name}.model_name_or_path is required.")
    adapter = payload.get("adapter_path")
    adapter_value = str(adapter).strip() if adapter else None
    return ModelEndpointConfig(
        model_name_or_path=model_reference,
        adapter_path=adapter_value,
        trust_remote_code=bool(payload.get("trust_remote_code", False)),
        local_files_only=bool(payload.get("local_files_only", False)),
        load_in_4bit=bool(payload.get("load_in_4bit", False)),
    )


def load_qwen_pair_config(path: str | Path) -> QwenPairConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Runtime config not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise TypeError("Runtime config must be a YAML mapping.")
    raw = _expand(raw)

    models = raw.get("models")
    if not isinstance(models, dict):
        raise KeyError("Missing config section: models")
    runtime = raw.get("runtime") or {}
    generation = raw.get("generation") or {}
    grounding = raw.get("grounding") or {}

    engine = str(generation.get("engine", "target")).strip().casefold()
    if engine not in {"target", "draft", "speculative"}:
        raise ValueError("generation.engine must be target, draft, or speculative.")

    max_new_tokens = int(generation.get("max_new_tokens", 256))
    draft_tokens = int(generation.get("draft_tokens_per_round", 4))
    if max_new_tokens <= 0:
        raise ValueError("generation.max_new_tokens must be positive.")
    if draft_tokens <= 0:
        raise ValueError("generation.draft_tokens_per_round must be positive.")

    prompt_mode = str(grounding.get("prompt_mode", "evidence_only")).strip().casefold()
    evidence_policy = str(grounding.get("evidence_policy", "compact")).strip().casefold()
    max_evidence_sources = int(grounding.get("max_evidence_sources", 6))
    max_evidence_characters = int(grounding.get("max_evidence_characters", 14_000))
    max_repair_attempts = int(grounding.get("max_repair_attempts", 1))
    max_answer_chars = int(grounding.get("max_answer_chars", 6000))

    if prompt_mode not in {"evidence_only", "scaffolded"}:
        raise ValueError("grounding.prompt_mode must be evidence_only or scaffolded.")
    if evidence_policy not in {"compact", "full", "structured_only", "guide_only"}:
        raise ValueError(
            "grounding.evidence_policy must be compact, full, structured_only, or guide_only."
        )
    if max_evidence_sources <= 0:
        raise ValueError("grounding.max_evidence_sources must be positive.")
    if max_evidence_characters <= 0:
        raise ValueError("grounding.max_evidence_characters must be positive.")
    if max_repair_attempts < 0 or max_repair_attempts > 2:
        raise ValueError("grounding.max_repair_attempts must be between 0 and 2.")
    if max_answer_chars <= 0:
        raise ValueError("grounding.max_answer_chars must be positive.")

    return QwenPairConfig(
        draft=_endpoint(models.get("draft"), "draft"),
        target=_endpoint(models.get("target"), "target"),
        device=str(runtime.get("device", "auto")),
        dtype=str(runtime.get("dtype", "auto")),
        generation=GenerationConfig(
            engine=engine,
            max_new_tokens=max_new_tokens,
            draft_tokens_per_round=draft_tokens,
            enable_thinking=bool(generation.get("enable_thinking", False)),
        ),
        grounding=GroundingConfig(
            require_citations=bool(grounding.get("require_citations", True)),
            fallback_on_error=bool(grounding.get("fallback_on_error", True)),
            max_answer_chars=max_answer_chars,
            prompt_mode=prompt_mode,
            evidence_policy=evidence_policy,
            max_evidence_sources=max_evidence_sources,
            max_evidence_characters=max_evidence_characters,
            max_repair_attempts=max_repair_attempts,
        ),
    )
