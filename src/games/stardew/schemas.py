"""Schemas shared by the Stardew Valley knowledge module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class PlayerState:
    game_version: str | None = None
    platform: str | None = None
    season: str | None = None
    day: int | None = None
    year: int | None = None
    weather: str | None = None
    time: str | None = None
    location: str | None = None
    route: str | None = None
    bundle_mode: str | None = None
    farm_type: str | None = None
    skills: dict[str, int] = field(default_factory=dict)
    friendship_hearts: dict[str, int] = field(default_factory=dict)
    budget: int | None = None
    goal: str | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "PlayerState":
        if value is None:
            return cls()
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: item for key, item in value.items() if key in allowed})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StardewRoute:
    intent: str
    entity: str | None
    confidence: float
    language: str
    player_state: PlayerState = field(default_factory=PlayerState)
    missing_context: list[str] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)

    @property
    def needs_context(self) -> bool:
        return bool(self.missing_context)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["player_state"] = self.player_state.to_dict()
        payload["needs_context"] = self.needs_context
        return payload
