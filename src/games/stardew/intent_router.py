"""Deterministic bilingual router for Stardew Valley fact and guide questions."""

from __future__ import annotations

import re
from typing import Any

from .query_store import StardewQueryStore
from .schemas import PlayerState, StardewRoute


class StardewIntentRouter:
    GUIDE_PATTERNS = (
        "what should i do", "how should i", "guide", "strategy", "getting started",
        "first spring", "community center", "skull cavern", "ginger island",
        "应该做什么", "怎么玩", "攻略", "开局", "第一年", "优先", "矿洞",
        "社区中心", "姜岛", "沙漠矿洞",
    )

    def __init__(self, store: StardewQueryStore) -> None:
        self.store = store

    @staticmethod
    def detect_language(question: str) -> str:
        return "zh" if re.search(r"[\u4e00-\u9fff]", question) else "en"

    @staticmethod
    def _extract_day(question: str) -> int | None:
        patterns = (
            r"(?:day|on)\s*(\d{1,2})",
            r"(?:第)?\s*(\d{1,2})\s*天",
        )
        for pattern in patterns:
            match = re.search(pattern, question, flags=re.IGNORECASE)
            if match:
                value = int(match.group(1))
                if 1 <= value <= 28:
                    return value
        return None


    @staticmethod
    def _extract_unknown_entity(question: str, intent: str) -> str | None:
        patterns = {
            "crop_info": (
                r"how long does (?P<entity>.+?) take to grow\??$",
                r"how many days does (?P<entity>.+?) take\??$",
                r"(?P<entity>.+?)需要几天成熟[？?]?$",
            ),
            "crop_deadline": (
                r"can i (?:still )?plant (?P<entity>.+?) in time\??$",
                r"can i harvest (?P<entity>.+?) if .*",
                r"(?P<entity>.+?)还来得及种[吗么？?]*$",
            ),
            "fish_availability": (
                r"where (?:and when )?can i catch (?P<entity>.+?)\??$",
                r"when can i catch (?P<entity>.+?)\??$",
                r"when can (?P<entity>.+?) be caught\??$",
                r"where is (?P<entity>.+?) available\??$",
                r"what conditions are required for (?P<entity>.+?)\??$",
                r"(?P<entity>.+?)(?:在哪里[、,，]?什么时候能钓|在哪里[、,，]?什么时候可以钓)[？?]?$",
                r"(?P<entity>.+?)在哪里钓[？?]?$",
                r"(?P<entity>.+?)什么时候能钓[？?]?$",
            ),
            "villager_gifts": (
                r"what gifts does (?P<entity>.+?) love\??$",
                r"what does (?P<entity>.+?) love\??$",
                r"(?P<entity>.+?)喜欢什么礼物[？?]?$",
            ),
            "recipe": (
                r"how do i (?:craft|make|cook) (?P<entity>.+?)\??$",
                r"(?:what is the recipe for|recipe for) (?P<entity>.+?)\??$",
                r"(?P<entity>.+?)怎么(?:做|制作|烹饪)[？?]?$",
                r"(?P<entity>.+?)的配方是什么[？?]?$",
            ),
            "acquisition": (
                r"where can i (?:get|buy|find|obtain) (?P<entity>.+?)\??$",
                r"how (?:do|can) i (?:get|obtain|find) (?P<entity>.+?)\??$",
                r"(?P<entity>.+?)(?:怎么获得|如何获得|哪里买|在哪买|从哪里获得)[？?]?$",
            ),
        }
        for pattern in patterns.get(intent, ()):
            match = re.search(pattern, question.strip(), flags=re.IGNORECASE)
            if match:
                entity = match.group("entity").strip(" ?？.。")
                return entity or None
        return None

    @staticmethod
    def _extract_state(question: str, supplied: dict[str, Any] | None) -> PlayerState:
        state = PlayerState.from_mapping(supplied)
        folded = question.casefold()
        season_terms = {
            "spring": ("spring", "春", "春季"),
            "summer": ("summer", "夏", "夏季"),
            "fall": ("fall", "autumn", "秋", "秋季"),
            "winter": ("winter", "冬", "冬季"),
        }
        if state.season is None:
            for season, terms in season_terms.items():
                if any(term in folded for term in terms):
                    state.season = season
                    break
        if state.day is None:
            state.day = StardewIntentRouter._extract_day(question)
        weather_terms = {
            "rain": ("rain", "rainy", "下雨", "雨天"),
            "sunny": ("sunny", "晴天", "晴"),
        }
        if state.weather is None:
            for weather, terms in weather_terms.items():
                if any(term in folded for term in terms):
                    state.weather = weather
                    break
        if state.bundle_mode is None:
            if "remixed" in folded or "混合收集包" in folded or "重混" in folded:
                state.bundle_mode = "remixed"
            elif "standard" in folded or "标准收集包" in folded:
                state.bundle_mode = "standard"
        return state

    def route(
        self,
        question: str,
        *,
        player_state: dict[str, Any] | None = None,
    ) -> StardewRoute:
        text = str(question).strip()
        if not text:
            raise ValueError("Stardew question cannot be empty.")
        folded = text.casefold()
        language = self.detect_language(text)
        state = self._extract_state(text, player_state)
        entity = self.store.resolve_entity_in_text(text)
        entity_name = entity["name"] if entity else None
        entity_type = entity["record_type"] if entity else None
        reason_codes: list[str] = []

        crop_deadline_pattern = any(
            term in folded
            for term in (
                "来得及",
                "in time",
                "harvest by",
                "plant on",
                "planted on",
                "if planted",
                "if i plant",
                "planting day",
                "before season ends",
                "还能收获",
                "能不能收获",
                "可以收获吗",
            )
        )
        fish_pattern = any(term in folded for term in (
            "where can i catch", "when can i catch", "where is", "when is",
            "what conditions", "available", "weather", "catch", "caught",
            "钓", "什么时候能钓", "在哪里钓", "哪里钓", "天气",
        ))
        gift_pattern = any(term in folded for term in ("gift", "love", "favorite", "喜欢", "礼物", "最爱"))
        crop_info_pattern = any(term in folded for term in ("take to grow", "days to grow", "mature", "crop", "几天成熟", "作物"))
        acquisition_pattern = any(term in folded for term in ("where can i get", "where can i buy", "where can i find", "how do i get", "how can i get", "obtain", "怎么获得", "如何获得", "哪里买", "在哪买", "从哪里获得"))
        reverse_bundle_pattern = any(term in folded for term in ("which bundle", "what bundle", "requires this", "哪个收集包", "哪个献祭包", "需要这个物品"))
        planting_recommendation_pattern = any(term in folded for term in ("what should i plant", "which crop should i plant", "what crop should", "今天种什么", "应该种什么", "种哪种作物"))
        cooking_progression_pattern = any(term in folded for term in (
            "unlock more cooking recipes", "learn more cooking recipes", "get more cooking recipes",
            "解锁更多烹饪配方", "学习更多烹饪配方",
        ))

        # For explicit question templates, resolve the extracted entity as a
        # complete name. This prevents a known entity such as ``Catfish`` from
        # being matched inside a false-premise name such as ``Galaxy Catfish``.
        extracted_by_intent = {
            candidate_intent: self._extract_unknown_entity(text, candidate_intent)
            for candidate_intent in (
                "crop_info",
                "crop_deadline",
                "fish_availability",
                "villager_gifts",
                "recipe",
                "acquisition",
            )
        }

        def exact_extracted(intent_name: str, record_type: str) -> tuple[str | None, str | None]:
            extracted = extracted_by_intent.get(intent_name)
            if not extracted:
                return None, None
            lookup = self.store.get_entity(extracted, record_type=record_type)
            if lookup.get("status") == "found":
                match = lookup["match"]
                return str(match["name"]), str(match["record_type"])
            return extracted, None

        if acquisition_pattern:
            extracted = extracted_by_intent.get("acquisition")
            if extracted:
                lookup = self.store.get_acquisition(extracted)
                if lookup.get("status") == "found":
                    match = lookup["match"]
                    entity_name, entity_type = str((match.get("facts") or {}).get("entity_name") or match["name"]), "acquisition"
                else:
                    entity_name, entity_type = extracted, None
        elif crop_deadline_pattern:
            extracted_name, extracted_type = exact_extracted("crop_deadline", "crop")
            if extracted_name is not None:
                entity_name, entity_type = extracted_name, extracted_type
        elif fish_pattern:
            extracted_name, extracted_type = exact_extracted("fish_availability", "fish")
            if extracted_name is not None:
                entity_name, entity_type = extracted_name, extracted_type
        elif gift_pattern:
            extracted_name, extracted_type = exact_extracted("villager_gifts", "villager")
            if extracted_name is not None:
                entity_name, entity_type = extracted_name, extracted_type
        elif crop_info_pattern:
            extracted_name, extracted_type = exact_extracted("crop_info", "crop")
            if extracted_name is not None:
                entity_name, entity_type = extracted_name, extracted_type

        if planting_recommendation_pattern:
            intent = "guide"
            entity_name = None
            missing = [field for field in ("season", "day") if getattr(state, field) is None]
            reason_codes.append("planting_recommendation_pattern")
        elif cooking_progression_pattern:
            intent = "guide"
            entity_name = None
            missing = []
            reason_codes.append("cooking_progression_pattern")
        elif acquisition_pattern:
            intent = "acquisition"
            entity_name = extracted_by_intent.get(intent) or entity_name or self._extract_unknown_entity(text, intent)
            missing = []
            reason_codes.append("acquisition_pattern")
        elif reverse_bundle_pattern and entity_name:
            intent = "bundles_requiring_item"
            missing = []
            reason_codes.append("reverse_bundle_pattern")
        elif (entity_type == "crop" and crop_deadline_pattern) or (entity_type is None and crop_deadline_pattern):
            intent = "crop_deadline"
            entity_name = extracted_by_intent.get(intent) or entity_name or self._extract_unknown_entity(text, intent)
            missing = [field for field in ("season", "day") if getattr(state, field) is None]
            reason_codes.append("crop_deadline_pattern")
        elif (entity_type == "fish" and fish_pattern) or (entity_type is None and any(term in folded for term in ("catch", "钓"))):
            intent = "fish_availability"
            entity_name = extracted_by_intent.get(intent) or entity_name or self._extract_unknown_entity(text, intent)
            current = any(term in folded for term in ("now", "right now", "today", "现在", "今天"))
            missing = [field for field in ("season", "weather", "time", "location") if current and getattr(state, field) is None]
            reason_codes.append("fish_availability_pattern")
        elif (entity_type == "villager" and gift_pattern) or (entity_type is None and gift_pattern):
            intent = "villager_gifts"
            entity_name = extracted_by_intent.get(intent) or entity_name or self._extract_unknown_entity(text, intent)
            missing = []
            reason_codes.append("villager_gift_pattern")
        elif entity_type is None and crop_info_pattern:
            intent = "crop_info"
            entity_name = self._extract_unknown_entity(text, intent) or text
            missing = []
            reason_codes.append("unknown_crop_pattern")
        elif entity_type == "villager":
            intent = "villager_info"
            missing = []
            reason_codes.append("villager_entity")
        elif entity_type == "recipe" or any(term in folded for term in ("recipe", "craft", "how do i make", "how do i cook", "怎么做", "怎么制作", "怎么烹饪", "配方")):
            intent = "recipe"
            entity_name = extracted_by_intent.get(intent) or entity_name or self._extract_unknown_entity(text, intent) or text
            missing = []
            reason_codes.append("recipe_pattern")
        elif entity_type == "bundle":
            intent = "bundle"
            missing = []
            reason_codes.append("bundle_entity")
        elif entity_type == "crop":
            intent = "crop_info"
            missing = []
            reason_codes.append("crop_entity")
        elif any(term in folded for term in self.GUIDE_PATTERNS):
            intent = "guide"
            missing = []
            reason_codes.append("guide_pattern")
        elif entity_name:
            intent = "entity"
            missing = []
            reason_codes.append("entity_match")
        else:
            intent = "guide" if len(text.split()) > 3 or language == "zh" else "search"
            missing = []
            reason_codes.append("fallback")

        return StardewRoute(
            intent=intent,
            entity=entity_name,
            confidence=0.95 if entity_name else 0.68,
            language=language,
            player_state=state,
            missing_context=missing,
            reason_codes=reason_codes,
        )
