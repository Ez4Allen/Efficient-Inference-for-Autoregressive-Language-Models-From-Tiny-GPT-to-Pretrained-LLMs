"""Deterministic Stardew Valley fact service with player-state conditions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .database_builder import DEFAULT_DATABASE_PATH
from .normalizers import normalize_season, normalize_weather, parse_clock
from .query_store import StardewQueryStore
from .schemas import PlayerState


class StardewFactService:
    VALID_INTENTS = {
        "entity",
        "crop_info",
        "crop_deadline",
        "fish_availability",
        "villager_info",
        "villager_gifts",
        "recipe",
        "recipes_using_item",
        "bundle",
        "bundles_requiring_item",
        "acquisition",
        "search",
    }

    def __init__(self, database_path: str | Path = DEFAULT_DATABASE_PATH) -> None:
        self.store = StardewQueryStore(database_path)
        self._closed = False

    def __enter__(self) -> "StardewFactService":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.store.close()
            self._closed = True

    @staticmethod
    def _provenance(record: dict[str, Any]) -> list[dict[str, Any]]:
        source = dict(record.get("provenance") or {})
        return [
            {
                "game": "stardew_valley",
                "entity_type": record.get("record_type"),
                "source_catalog_id": record.get("source_catalog_id"),
                "entity_name": record.get("name"),
                "game_version": record.get("game_version"),
                "platform": record.get("platform"),
                **source,
            }
        ]

    @staticmethod
    def _base(
        *,
        status: str,
        intent: str,
        query: str,
        entity: str | None,
        facts: Any,
        warnings: list[str] | None = None,
        candidates: list[dict[str, Any]] | None = None,
        provenance: list[dict[str, Any]] | None = None,
        missing_context: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "game": "stardew_valley",
            "status": status,
            "intent": intent,
            "query": query,
            "entity": entity,
            "facts": facts,
            "warnings": warnings or [],
            "candidates": candidates or [],
            "provenance": provenance or [],
            "missing_context": missing_context or [],
        }

    def _single(
        self,
        result: dict[str, Any],
        *,
        intent: str,
        query: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if result["status"] == "not_found":
            return None, self._base(
                status="not_found",
                intent=intent,
                query=query,
                entity=query,
                facts=None,
                warnings=["No matching Stardew Valley fact was found in the local snapshot."],
            )
        if result["status"] == "ambiguous":
            candidates = [
                {
                    "name": item["name"],
                    "record_type": item["record_type"],
                    "source_catalog_id": item["source_catalog_id"],
                }
                for item in result["matches"]
            ]
            return None, self._base(
                status="ambiguous",
                intent=intent,
                query=query,
                entity=None,
                facts=None,
                warnings=["Multiple Stardew Valley entities match this name."],
                candidates=candidates,
                provenance=[entry for item in result["matches"] for entry in self._provenance(item)],
            )
        return result["match"], None

    def entity(self, name: str) -> dict[str, Any]:
        record, error = self._single(self.store.get_entity(name), intent="entity", query=name)
        if error:
            return error
        assert record is not None
        return self._base(
            status="found", intent="entity", query=name, entity=record["name"],
            facts={"record_type": record["record_type"], **record["facts"], "conditions": record["conditions"]},
            provenance=self._provenance(record),
        )

    def crop_info(self, name: str) -> dict[str, Any]:
        record, error = self._single(self.store.get_crop(name), intent="crop_info", query=name)
        if error:
            return error
        assert record is not None
        return self._base(
            status="found", intent="crop_info", query=name, entity=record["name"],
            facts=record["facts"], provenance=self._provenance(record),
        )

    def crop_deadline(
        self,
        name: str,
        *,
        player_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = PlayerState.from_mapping(player_state)
        missing = [field for field in ("season", "day") if getattr(state, field) is None]
        if missing:
            return self._base(
                status="needs_context", intent="crop_deadline", query=name, entity=name,
                facts=None, warnings=["Crop deadline calculation requires season and calendar day."],
                missing_context=missing,
            )
        season = normalize_season(state.season)
        day = int(state.day)
        if not 1 <= day <= 28:
            raise ValueError("Stardew calendar day must be between 1 and 28.")
        record, error = self._single(self.store.get_crop(name), intent="crop_deadline", query=name)
        if error:
            return error
        assert record is not None
        facts = record["facts"]
        valid_seasons = facts.get("seasons") or []
        growth_days = int(facts["growth_days"])
        harvest_day = day + growth_days
        in_season = season in valid_seasons
        can_harvest = in_season and harvest_day <= 28
        latest_planting_day = 28 - growth_days
        regrow_days = facts.get("regrow_days")
        harvests = 0
        if can_harvest:
            harvests = 1
            if regrow_days:
                harvests += max(0, (28 - harvest_day) // int(regrow_days))
        output = {
            "crop_name": record["name"],
            "season": season,
            "planting_day": day,
            "growth_days": growth_days,
            "first_harvest_day": harvest_day if can_harvest else None,
            "latest_planting_day": latest_planting_day,
            "can_harvest_before_season_end": can_harvest,
            "estimated_harvests_before_season_end": harvests,
            "regrow_days": regrow_days,
        }
        warnings = []
        if not in_season:
            warnings.append(f"{record['name']} is not a normal {season} crop.")
        elif not can_harvest:
            warnings.append("The crop would mature after day 28 without growth-speed modifiers.")
        return self._base(
            status="found", intent="crop_deadline", query=name, entity=record["name"],
            facts=output, warnings=warnings, provenance=self._provenance(record),
        )

    @staticmethod
    def _window_matches(window: dict[str, Any], state: PlayerState) -> bool:
        if state.season is not None:
            season = normalize_season(state.season)
            if season not in (window.get("seasons") or []):
                return False
        if state.weather is not None:
            weather = normalize_weather(state.weather)
            allowed = set(window.get("weather") or [])
            # Thunderstorms satisfy ordinary rain requirements. A Rain Totem
            # also produces rain-like fishing conditions while remaining
            # separately representable for winter-only windows.
            observed_conditions = {weather}
            if weather == "storm":
                observed_conditions.add("rain")
            elif weather == "rain_totem":
                observed_conditions.add("rain")
            if "any" not in allowed and not (allowed & observed_conditions):
                return False
        if state.location is not None:
            location = str(state.location).strip().casefold().replace(" ", "_")
            if location not in {str(value).casefold().replace(" ", "_") for value in window.get("locations") or []}:
                return False
        if state.time is not None:
            current = parse_clock(state.time)
            start = parse_clock(window.get("time_start"))
            end = parse_clock(window.get("time_end"))
            # Stardew represents post-midnight windows as 24:00-26:00.
            # Only shift an early-morning query into that range when the
            # availability window itself crosses midnight. All-day windows
            # such as 00:00-24:00 must continue to match 01:00 normally.
            if (
                current is not None
                and end is not None
                and end > 1440
                and current < 360
            ):
                current += 1440
            if start is not None and end is not None and not (start <= current <= end):
                return False
        return True

    def fish_availability(
        self,
        name: str,
        *,
        player_state: dict[str, Any] | None = None,
        require_current_state: bool = False,
    ) -> dict[str, Any]:
        state = PlayerState.from_mapping(player_state)
        if require_current_state:
            missing = [field for field in ("season", "weather", "time", "location") if getattr(state, field) is None]
            if missing:
                return self._base(
                    status="needs_context", intent="fish_availability", query=name, entity=name,
                    facts=None, warnings=["Checking whether a fish is catchable now requires season, weather, time, and location."],
                    missing_context=missing,
                )
        record, error = self._single(self.store.get_fish(name), intent="fish_availability", query=name)
        if error:
            return error
        assert record is not None
        windows = list(record["facts"].get("availability_windows") or [])
        matching = [window for window in windows if self._window_matches(window, state)]
        output = {
            "fish_name": record["name"],
            "difficulty": record["facts"].get("difficulty"),
            "behavior": record["facts"].get("behavior"),
            "availability_windows": windows,
            "matching_windows": matching if player_state else windows,
            "catchable_for_player_state": bool(matching) if player_state else None,
            "player_state": state.to_dict(),
        }
        return self._base(
            status="found", intent="fish_availability", query=name, entity=record["name"],
            facts=output, provenance=self._provenance(record),
        )

    def villager_info(self, name: str) -> dict[str, Any]:
        record, error = self._single(self.store.get_villager(name), intent="villager_info", query=name)
        if error:
            return error
        assert record is not None
        return self._base(
            status="found", intent="villager_info", query=name, entity=record["name"],
            facts=record["facts"], provenance=self._provenance(record),
        )

    def villager_gifts(self, name: str) -> dict[str, Any]:
        result = self.villager_info(name)
        result["intent"] = "villager_gifts"
        if result["status"] == "found":
            result["facts"] = {
                "villager_name": result["entity"],
                "loved_gifts": result["facts"].get("loved_gifts") or [],
                "birthday": result["facts"].get("birthday"),
            }
        return result

    def recipe(self, name: str) -> dict[str, Any]:
        record, error = self._single(self.store.get_recipe(name), intent="recipe", query=name)
        if error:
            return error
        assert record is not None
        return self._base(
            status="found", intent="recipe", query=name, entity=record["name"],
            facts=record["facts"], provenance=self._provenance(record),
        )

    def recipes_using_item(self, item_name: str) -> dict[str, Any]:
        records = self.store.recipes_using_item(item_name)
        return self._base(
            status="found" if records else "not_found", intent="recipes_using_item",
            query=item_name, entity=item_name,
            facts={"item_name": item_name, "recipes": [record["name"] for record in records]},
            warnings=[] if records else ["No tracked recipe uses this ingredient."],
            provenance=[entry for record in records for entry in self._provenance(record)],
        )

    def bundle(
        self,
        name: str,
        *,
        bundle_mode: str = "standard",
    ) -> dict[str, Any]:
        lookup = self.store.get_bundle(name, bundle_mode=bundle_mode)
        if lookup.get("status") == "not_found" and str(bundle_mode).casefold() == "remixed":
            standard = self.store.get_bundle(name, bundle_mode="standard")
            if standard.get("status") == "found":
                record = standard["match"]
                return self._base(
                    status="partial",
                    intent="bundle",
                    query=name,
                    entity=record["name"],
                    facts={
                        "requested_bundle_mode": "remixed",
                        "available_bundle_mode": "standard",
                        "standard_reference": record["facts"],
                    },
                    warnings=[
                        "The curated snapshot has complete Standard Bundle coverage but does not claim complete Remixed Bundle coverage."
                    ],
                    provenance=self._provenance(record),
                )
        record, error = self._single(lookup, intent="bundle", query=name)
        if error:
            return error
        assert record is not None
        return self._base(
            status="found", intent="bundle", query=name, entity=record["name"],
            facts=record["facts"], provenance=self._provenance(record),
        )

    def bundles_requiring_item(self, item_name: str, *, bundle_mode: str = "standard") -> dict[str, Any]:
        records = self.store.bundles_requiring_item(item_name, bundle_mode=bundle_mode)
        return self._base(
            status="found" if records else "not_found", intent="bundles_requiring_item",
            query=item_name, entity=item_name,
            facts={"item_name": item_name, "bundle_mode": bundle_mode, "bundles": [record["name"] for record in records]},
            warnings=[] if records else ["No tracked bundle requires this item under the selected bundle mode."],
            provenance=[entry for record in records for entry in self._provenance(record)],
        )


    def acquisition(self, name: str) -> dict[str, Any]:
        result = self.store.get_acquisition(name)
        if result.get("status") == "not_found":
            return self._base(
                status="not_found",
                intent="acquisition",
                query=name,
                entity=name,
                facts=None,
                warnings=["No tracked acquisition source was found in the curated snapshot."],
            )
        records = list(result.get("matches") or [])
        if not records:
            return self._base(
                status="not_found", intent="acquisition", query=name,
                entity=name, facts=None,
            )
        sources: list[dict[str, Any]] = []
        for record in records:
            sources.extend(list((record.get("facts") or {}).get("sources") or []))
        canonical = str((records[0].get("facts") or {}).get("entity_name") or records[0]["name"])
        return self._base(
            status="found",
            intent="acquisition",
            query=name,
            entity=canonical,
            facts={
                "entity_name": canonical,
                "entity_type": (records[0].get("facts") or {}).get("entity_type"),
                "sources": sources,
            },
            provenance=[entry for record in records for entry in self._provenance(record)],
        )

    def search(self, query: str, *, limit: int = 10) -> dict[str, Any]:
        records = self.store.search(query, limit=limit)
        return self._base(
            status="found" if records else "not_found", intent="search", query=query,
            entity=None, facts={"matches": records}, provenance=[entry for record in records for entry in self._provenance(record)],
        )

    def query(
        self,
        intent: str,
        entity: str,
        *,
        player_state: dict[str, Any] | None = None,
        limit: int = 10,
        bundle_mode: str = "standard",
        require_current_state: bool = False,
    ) -> dict[str, Any]:
        normalized = str(intent).strip().casefold()
        if normalized not in self.VALID_INTENTS:
            raise ValueError(f"Unsupported Stardew intent: {intent!r}")
        if normalized == "entity":
            return self.entity(entity)
        if normalized == "crop_info":
            return self.crop_info(entity)
        if normalized == "crop_deadline":
            return self.crop_deadline(entity, player_state=player_state)
        if normalized == "fish_availability":
            return self.fish_availability(entity, player_state=player_state, require_current_state=require_current_state)
        if normalized == "villager_info":
            return self.villager_info(entity)
        if normalized == "villager_gifts":
            return self.villager_gifts(entity)
        if normalized == "recipe":
            return self.recipe(entity)
        if normalized == "recipes_using_item":
            return self.recipes_using_item(entity)
        if normalized == "bundle":
            return self.bundle(entity, bundle_mode=bundle_mode)
        if normalized == "bundles_requiring_item":
            return self.bundles_requiring_item(entity, bundle_mode=bundle_mode)
        if normalized == "acquisition":
            return self.acquisition(entity)
        return self.search(entity, limit=limit)
