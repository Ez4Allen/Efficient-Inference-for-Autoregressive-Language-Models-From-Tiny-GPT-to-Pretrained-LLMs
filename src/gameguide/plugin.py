"""Protocol implemented by each game knowledge plug-in."""

from __future__ import annotations

from typing import Any, Protocol

from .schemas import GameGuideResult


class GamePlugin(Protocol):
    game_id: str
    display_name: str

    def answer(
        self,
        question: str,
        *,
        language: str = "auto",
        player_state: dict[str, Any] | None = None,
        include_debug: bool = False,
    ) -> GameGuideResult: ...

    def close(self) -> None: ...
