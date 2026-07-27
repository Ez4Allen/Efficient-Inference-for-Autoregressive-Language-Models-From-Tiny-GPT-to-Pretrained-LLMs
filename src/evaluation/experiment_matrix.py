
"""Declarative ablation matrix for GameGuideLM model experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ExperimentCondition:
    name: str
    generator: str
    engine: str = "target"
    prompt_mode: str = "evidence_only"
    evidence_policy: str = "compact"
    description: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Experiment condition names cannot be empty.")
        if self.generator not in {"deterministic", "grounded", "ungrounded"}:
            raise ValueError(
                "generator must be deterministic, grounded, or ungrounded."
            )
        if self.engine not in {"target", "draft", "speculative"}:
            raise ValueError("engine must be target, draft, or speculative.")
        if self.prompt_mode not in {"evidence_only", "scaffolded"}:
            raise ValueError("prompt_mode must be evidence_only or scaffolded.")
        if self.evidence_policy not in {
            "compact",
            "full",
            "structured_only",
            "guide_only",
        }:
            raise ValueError("Unsupported evidence policy.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ExperimentMatrix:
    version: int
    conditions: tuple[ExperimentCondition, ...]

    def __post_init__(self) -> None:
        if self.version <= 0:
            raise ValueError("Experiment matrix version must be positive.")
        if not self.conditions:
            raise ValueError("Experiment matrix requires at least one condition.")
        names = [condition.name for condition in self.conditions]
        if len(names) != len(set(names)):
            raise ValueError("Experiment condition names must be unique.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "conditions": [condition.to_dict() for condition in self.conditions],
        }


def load_experiment_matrix(path: str | Path) -> ExperimentMatrix:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Experiment matrix not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Experiment matrix must be a YAML mapping.")
    raw_conditions = payload.get("conditions")
    if not isinstance(raw_conditions, list):
        raise TypeError("Experiment matrix conditions must be a list.")
    conditions = []
    for raw in raw_conditions:
        if not isinstance(raw, dict):
            raise TypeError("Each experiment condition must be a mapping.")
        conditions.append(
            ExperimentCondition(
                name=str(raw.get("name", "")).strip(),
                generator=str(raw.get("generator", "deterministic")).strip().casefold(),
                engine=str(raw.get("engine", "target")).strip().casefold(),
                prompt_mode=str(raw.get("prompt_mode", "evidence_only")).strip().casefold(),
                evidence_policy=str(raw.get("evidence_policy", "compact")).strip().casefold(),
                description=str(raw.get("description", "")).strip(),
            )
        )
    return ExperimentMatrix(
        version=int(payload.get("version", 1)),
        conditions=tuple(conditions),
    )
