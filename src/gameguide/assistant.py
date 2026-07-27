"""Game-agnostic orchestration over registered knowledge plug-ins."""

from __future__ import annotations

from typing import Any

from .plugin import GamePlugin
from .schemas import GameGuideResult


class GameGuideAssistant:
    def __init__(self, plugins: list[GamePlugin], *, generator: Any | None = None) -> None:
        if not plugins:
            raise ValueError("GameGuideAssistant requires at least one game plug-in.")
        plugin_map: dict[str, GamePlugin] = {}
        for plugin in plugins:
            game_id = str(plugin.game_id).strip().casefold().replace(" ", "_")
            if not game_id:
                raise ValueError("Game plug-in IDs cannot be empty.")
            if game_id in plugin_map:
                raise ValueError(f"Duplicate game plug-in ID: {game_id}")
            plugin_map[game_id] = plugin
        self.plugins = plugin_map
        self.generator = generator
        self._closed = False

    def __enter__(self) -> "GameGuideAssistant":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("GameGuideAssistant is closed.")

    def close(self) -> None:
        if self._closed:
            return
        for plugin in self.plugins.values():
            plugin.close()
        if self.generator is not None and hasattr(self.generator, "close"):
            self.generator.close()
        self._closed = True

    @property
    def available_games(self) -> tuple[str, ...]:
        return tuple(sorted(self.plugins))

    def answer(
        self,
        question: str,
        *,
        game: str,
        language: str = "auto",
        player_state: dict[str, Any] | None = None,
        include_debug: bool = False,
    ) -> GameGuideResult:
        self._ensure_open()
        normalized_question = str(question).strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty.")
        game_id = str(game).strip().casefold().replace(" ", "_")
        aliases = {
            "stardew": "stardew_valley",
            "sdv": "stardew_valley",
            "stardew_valley": "stardew_valley",
            "terraria": "terraria",
        }
        game_id = aliases.get(game_id, game_id)
        if game_id not in self.plugins:
            raise ValueError(f"Unsupported game {game!r}. Available: {sorted(self.plugins)}")
        result = self.plugins[game_id].answer(
            normalized_question,
            language=language,
            player_state=player_state,
            include_debug=include_debug,
        )
        if result.game != game_id:
            raise RuntimeError(
                f"Plug-in {game_id!r} returned a result for game {result.game!r}."
            )
        if self.generator is not None:
            result.answer = self.generator.generate(result)
            result.warnings = list(
                dict.fromkeys([*result.warnings, *getattr(self.generator, "last_warnings", [])])
            )
            if include_debug:
                result.debug["generation"] = dict(
                    getattr(self.generator, "last_debug", {}) or {}
                )
        return result
