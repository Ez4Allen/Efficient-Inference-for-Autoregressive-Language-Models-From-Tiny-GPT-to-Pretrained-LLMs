"""Deterministic bilingual intent routing for Terraria questions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Pattern

from .schemas import AssistantIntent, AssistantRequest, RouteDecision


_CJK_PATTERN = re.compile(r"[\u3400-\u9fff]")
_TRAILING_CONTEXT = re.compile(
    r"\s+(?:in\s+terraria|for\s+terraria|在泰拉瑞亚(?:中|里)?)\s*$",
    flags=re.IGNORECASE,
)
_MODE_PATTERNS = {
    "master": re.compile(r"\bmaster(?:\s+mode)?\b|大师(?:模式)?", re.IGNORECASE),
    "expert": re.compile(r"\bexpert(?:\s+mode)?\b|专家(?:模式)?", re.IGNORECASE),
    "normal": re.compile(r"\bnormal(?:\s+mode)?\b|普通(?:模式)?", re.IGNORECASE),
}
_ITEM_ID_PATTERN = re.compile(r"\bitem\s*id\s*[:=#]?\s*(-?\d+)\b", re.IGNORECASE)
_NPC_ID_PATTERN = re.compile(r"\bnpc\s*id\s*[:=#]?\s*(-?\d+)\b", re.IGNORECASE)
_INTERNAL_NAME_PATTERN = re.compile(
    r"\binternal(?:\s+name)?\s*[:=]\s*([A-Za-z0-9_]+)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RouteRule:
    intent: AssistantIntent
    pattern: Pattern[str]
    confidence: float
    reason_code: str


class IntentRouter:
    """Route common English and Chinese Terraria questions without an LLM."""

    def __init__(self) -> None:
        self.rules = self._build_rules()

    @staticmethod
    def detect_language(text: str) -> str:
        return "zh" if _CJK_PATTERN.search(text) else "en"

    @staticmethod
    def _build_rules() -> tuple[RouteRule, ...]:
        flags = re.IGNORECASE
        rules = [
            # Explicit command-style forms are useful in notebooks and APIs.
            RouteRule(
                AssistantIntent.RECIPE,
                re.compile(r"^\s*(?:recipe|craft)\s*[:：]\s*(?P<entity>.+?)\s*$", flags),
                0.99,
                "explicit_recipe_command",
            ),
            RouteRule(
                AssistantIntent.DROPS_FOR_ITEM,
                re.compile(r"^\s*(?:drop\s+sources?|where\s+from)\s*[:：]\s*(?P<entity>.+?)\s*$", flags),
                0.99,
                "explicit_item_drop_command",
            ),
            RouteRule(
                AssistantIntent.DROPS_FROM_SOURCE,
                re.compile(r"^\s*(?:drops?|loot)\s*[:：]\s*(?P<entity>.+?)\s*$", flags),
                0.99,
                "explicit_source_drop_command",
            ),
            RouteRule(
                AssistantIntent.ITEM,
                re.compile(r"^\s*item\s*[:：]\s*(?P<entity>.+?)\s*$", flags),
                0.99,
                "explicit_item_command",
            ),
            RouteRule(
                AssistantIntent.NPC,
                re.compile(r"^\s*(?:npc|boss)\s*[:：]\s*(?P<entity>.+?)\s*$", flags),
                0.99,
                "explicit_npc_command",
            ),
            # Recipe questions.
            RouteRule(
                AssistantIntent.RECIPE,
                re.compile(
                    r"^\s*(?:show\s+)?(?:all|legacy|old)\s+(?:recipes?|variants?)\s+(?:for|of)\s+(?P<entity>.+?)\s*[?!.]*$",
                    flags,
                ),
                0.97,
                "english_all_recipe_variants",
            ),
            RouteRule(
                AssistantIntent.RECIPE,
                re.compile(
                    r"^\s*(?:显示|查看)?\s*(?P<entity>.+?)\s*(?:的)?(?:全部|所有|旧版)配方\s*[？?。!！]*$",
                ),
                0.97,
                "chinese_all_recipe_variants",
            ),
            RouteRule(
                AssistantIntent.RECIPE,
                re.compile(
                    r"^\s*how\s+(?:do\s+i\s+|can\s+i\s+|to\s+)?(?:craft|make|create)\s+(?P<entity>.+?)\s*[?!.]*$",
                    flags,
                ),
                0.97,
                "english_recipe_question",
            ),
            RouteRule(
                AssistantIntent.RECIPE,
                re.compile(
                    r"^\s*(?:what(?:'s|\s+is)\s+the\s+)?recipe\s+(?:for|of)\s+(?P<entity>.+?)\s*[?!.]*$",
                    flags,
                ),
                0.97,
                "english_recipe_question",
            ),
            RouteRule(
                AssistantIntent.RECIPE,
                re.compile(r"^\s*(?P<entity>.+?)\s*(?:怎么|如何)(?:合成|制作|做)\s*[？?。!！]*$"),
                0.97,
                "chinese_recipe_question",
            ),
            RouteRule(
                AssistantIntent.RECIPE,
                re.compile(r"^\s*(?:怎么|如何)(?:合成|制作|做)\s*(?P<entity>.+?)\s*[？?。!！]*$"),
                0.97,
                "chinese_recipe_question",
            ),
            RouteRule(
                AssistantIntent.RECIPE,
                re.compile(r"^\s*(?P<entity>.+?)\s*(?:的)?配方(?:是什么)?\s*[？?。!！]*$"),
                0.96,
                "chinese_recipe_question",
            ),
            # Reverse recipe questions must precede ordinary item questions.
            RouteRule(
                AssistantIntent.RECIPES_USING_ITEM,
                re.compile(
                    r"^\s*what\s+(?:can\s+i\s+)?(?:craft|make)\s+(?:with|using)\s+(?P<entity>.+?)\s*[?!.]*$",
                    flags,
                ),
                0.96,
                "english_reverse_recipe_question",
            ),
            RouteRule(
                AssistantIntent.RECIPES_USING_ITEM,
                re.compile(
                    r"^\s*(?:what\s+uses|what\s+is)\s+(?P<entity>.+?)\s+(?:used\s+for|used\s+to\s+craft)\s*[?!.]*$",
                    flags,
                ),
                0.94,
                "english_reverse_recipe_question",
            ),
            RouteRule(
                AssistantIntent.RECIPES_USING_ITEM,
                re.compile(r"^\s*(?P<entity>.+?)\s*(?:能|可以)(?:合成|制作|做)(?:什么|哪些东西)\s*[？?。!！]*$"),
                0.96,
                "chinese_reverse_recipe_question",
            ),
            RouteRule(
                AssistantIntent.RECIPES_USING_ITEM,
                re.compile(r"^\s*(?P<entity>.+?)\s*(?:有什么用途|能做什么)\s*[？?。!！]*$"),
                0.92,
                "chinese_reverse_recipe_question",
            ),
            # Source -> drops.
            RouteRule(
                AssistantIntent.DROPS_FROM_SOURCE,
                re.compile(
                    r"^\s*what\s+(?:does|do)\s+(?P<entity>.+?)\s+drop(?:s)?\s*[?!.]*$",
                    flags,
                ),
                0.98,
                "english_source_drop_question",
            ),
            RouteRule(
                AssistantIntent.DROPS_FROM_SOURCE,
                re.compile(r"^\s*(?P<entity>.+?)\s*(?:会)?(?:掉|掉落)(?:什么|哪些东西)\s*[？?。!！]*$"),
                0.98,
                "chinese_source_drop_question",
            ),
            # Item -> sources.
            RouteRule(
                AssistantIntent.DROPS_FOR_ITEM,
                re.compile(
                    r"^\s*(?:where|who)\s+(?:does|do|can)\s+(?P<entity>.+?)\s+(?:drop|come\s+from)\s*[?!.]*$",
                    flags,
                ),
                0.97,
                "english_item_drop_question",
            ),
            RouteRule(
                AssistantIntent.DROPS_FOR_ITEM,
                re.compile(r"^\s*where\s+(?:can\s+i\s+)?(?:get|find|farm)\s+(?P<entity>.+?)\s*[?!.]*$", flags),
                0.96,
                "english_item_drop_question",
            ),
            RouteRule(
                AssistantIntent.DROPS_FOR_ITEM,
                re.compile(r"^\s*how\s+(?:do\s+i\s+|can\s+i\s+)?get\s+(?P<entity>.+?)\s*[?!.]*$", flags),
                0.93,
                "english_item_acquisition_question",
            ),
            RouteRule(
                AssistantIntent.DROPS_FOR_ITEM,
                re.compile(r"^\s*(?P<entity>.+?)\s*(?:从哪|哪里|在哪)(?:掉|掉落|获得|获取)\s*[？?。!！]*$"),
                0.97,
                "chinese_item_drop_question",
            ),
            RouteRule(
                AssistantIntent.DROPS_FOR_ITEM,
                re.compile(r"^\s*(?P<entity>.+?)\s*(?:怎么|如何)(?:获得|获取|刷)\s*[？?。!！]*$"),
                0.94,
                "chinese_item_acquisition_question",
            ),
            RouteRule(
                AssistantIntent.DROPS_FOR_ITEM,
                re.compile(r"^\s*(?:谁|什么怪)(?:会)?掉(?:落)?\s*(?P<entity>.+?)\s*[？?。!！]*$"),
                0.97,
                "chinese_item_drop_question",
            ),
            # NPC facts.
            RouteRule(
                AssistantIntent.NPC,
                re.compile(
                    r"^\s*(?:show|give|tell)\s+(?:me\s+)?(?:the\s+)?(?:stats?|information)\s+(?:for|about|of)\s+(?P<entity>.+?)\s*[?!.]*$",
                    flags,
                ),
                0.94,
                "english_npc_stats_question",
            ),
            RouteRule(
                AssistantIntent.NPC,
                re.compile(r"^\s*(?:what\s+are|show)\s+(?P<entity>.+?)\s+(?:stats?|statistics)\s*[?!.]*$", flags),
                0.94,
                "english_npc_stats_question",
            ),
            RouteRule(
                AssistantIntent.NPC,
                re.compile(r"^\s*(?P<entity>.+?)\s*(?:的)?(?:属性|血量|生命|防御|伤害|免疫)\s*[是什么多少？?。!！]*$"),
                0.94,
                "chinese_npc_stats_question",
            ),
            # Explicit item facts.
            RouteRule(
                AssistantIntent.ITEM,
                re.compile(r"^\s*(?:show|tell\s+me\s+about)\s+(?:the\s+)?item\s+(?P<entity>.+?)\s*[?!.]*$", flags),
                0.94,
                "english_item_fact_question",
            ),
            RouteRule(
                AssistantIntent.ITEM,
                re.compile(r"^\s*(?P<entity>.+?)\s*(?:物品)?(?:属性|信息)\s*[是什么多少？?。!！]*$"),
                0.90,
                "chinese_item_fact_question",
            ),
        ]
        return tuple(rules)

    @staticmethod
    def _clean_entity(value: str) -> str:
        entity = value.strip()
        entity = _TRAILING_CONTEXT.sub("", entity).strip()
        entity = _ITEM_ID_PATTERN.sub("", entity)
        entity = _NPC_ID_PATTERN.sub("", entity)
        entity = _INTERNAL_NAME_PATTERN.sub("", entity)
        entity = re.sub(
            r"^(?:the|an|a)\s+",
            "",
            entity,
            flags=re.IGNORECASE,
        )
        entity = re.sub(
            r"\s+(?:in|on)\s+(?:normal|expert|master)(?:\s+mode)?\s*$",
            "",
            entity,
            flags=re.IGNORECASE,
        )
        entity = re.sub(r"(?:普通|专家|大师)(?:模式)?\s*$", "", entity).strip()
        entity = entity.strip(" \t\r\n\"'“”‘’`?？!！.。,:：;")
        return entity

    @staticmethod
    def _extract_parameters(question: str) -> dict[str, object]:
        parameters: dict[str, object] = {}
        for mode, pattern in _MODE_PATTERNS.items():
            if pattern.search(question):
                parameters["mode"] = mode
                break

        item_id_match = _ITEM_ID_PATTERN.search(question)
        if item_id_match:
            parameters["item_id"] = int(item_id_match.group(1))

        npc_id_match = _NPC_ID_PATTERN.search(question)
        if npc_id_match:
            parameters["npc_id"] = int(npc_id_match.group(1))

        internal_match = _INTERNAL_NAME_PATTERN.search(question)
        if internal_match:
            parameters["internal_name"] = internal_match.group(1)

        if re.search(r"\b(?:all|legacy|old)\s+(?:recipes?|variants?)\b|全部(?:配方|方案)|旧版配方", question, re.IGNORECASE):
            parameters["preferred_only"] = False

        return parameters

    def route(self, request: AssistantRequest | str) -> RouteDecision:
        if isinstance(request, str):
            request = AssistantRequest(request)

        question = request.question
        parameters = self._extract_parameters(question)
        routing_question = re.sub(
            r"\b(?:in|on)\s+(?:normal|expert|master)(?:\s+mode)?\b\s*,?",
            "",
            question,
            flags=re.IGNORECASE,
        )
        routing_question = re.sub(
            r"(?:普通|专家|大师)(?:模式)?(?:下|中|里)?\s*[，,]?",
            "",
            routing_question,
        ).strip()

        for rule in self.rules:
            match = rule.pattern.match(routing_question)
            if not match:
                continue
            entity = self._clean_entity(match.group("entity"))
            if entity:
                return RouteDecision(
                    intent=rule.intent,
                    entity=entity,
                    confidence=rule.confidence,
                    parameters=parameters,
                    reason_codes=[rule.reason_code],
                )

        # Broad lookups are routed through catalog search so the resolver can
        # present Item/NPC/Recipe candidates rather than guessing a type.
        broad_patterns = (
            re.compile(r"^\s*(?:what|who)\s+is\s+(?P<entity>.+?)\s*[?!.]*$", re.IGNORECASE),
            re.compile(r"^\s*tell\s+me\s+about\s+(?P<entity>.+?)\s*[?!.]*$", re.IGNORECASE),
            re.compile(r"^\s*(?P<entity>.+?)\s*(?:是什么|介绍|资料)\s*[？?。!！]*$"),
        )
        for pattern in broad_patterns:
            match = pattern.match(routing_question)
            if match:
                entity = self._clean_entity(match.group("entity"))
                if entity:
                    return RouteDecision(
                        intent=AssistantIntent.SEARCH,
                        entity=entity,
                        confidence=0.78,
                        parameters=parameters,
                        reason_codes=["broad_entity_lookup"],
                    )

        entity = self._clean_entity(question)
        return RouteDecision(
            intent=AssistantIntent.SEARCH if entity else AssistantIntent.UNKNOWN,
            entity=entity or None,
            confidence=0.45 if entity else 0.0,
            parameters=parameters,
            reason_codes=["fallback_catalog_search" if entity else "empty_route"],
        )
