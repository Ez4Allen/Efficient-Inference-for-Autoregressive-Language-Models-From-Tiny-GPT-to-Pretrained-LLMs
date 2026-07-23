"""Deterministic grounded answer rendering for Terraria fact packages."""

from __future__ import annotations

from typing import Any

from .schemas import AssistantIntent, AssistantRequest, RouteDecision


def _join(values: list[str], *, language: str) -> str:
    cleaned = [str(value) for value in values if str(value).strip()]
    if not cleaned:
        return "无" if language == "zh" else "None"
    return "、".join(cleaned) if language == "zh" else ", ".join(cleaned)


def _coin_display(copper: int | None, *, language: str) -> str | None:
    if copper is None:
        return None
    platinum, remainder = divmod(int(copper), 1_000_000)
    gold, remainder = divmod(remainder, 10_000)
    silver, copper_value = divmod(remainder, 100)
    units = []
    labels = ("铂金币", "金币", "银币", "铜币") if language == "zh" else (
        "platinum",
        "gold",
        "silver",
        "copper",
    )
    for value, label in zip((platinum, gold, silver, copper_value), labels):
        if value:
            units.append(f"{value} {label}")
    return _join(units, language=language) if units else f"0 {labels[-1]}"


def _mode_value(values: Any, mode: str) -> Any:
    if not isinstance(values, dict):
        return None
    if mode in values:
        return values[mode]
    return values.get("all")


def _number_display(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        return "/".join(str(item) for item in value) if value else None
    if isinstance(value, dict):
        if "primary" in value:
            primary = value.get("primary")
            if primary is not None:
                return str(primary)
        variants = value.get("variants")
        if isinstance(variants, list) and variants:
            return "/".join(str(item) for item in variants)
        if "all" in value:
            return _number_display(value["all"])
        return None
    return str(value)


class DeterministicAnswerRenderer:
    """Render concise answers while preserving ambiguity and missing evidence."""

    def __init__(self, *, max_entries: int = 12) -> None:
        self.max_entries = max(1, int(max_entries))

    @staticmethod
    def _clarification(route: RouteDecision, language: str) -> str:
        if route.clarification_question:
            if language == "zh":
                names = [candidate.get("name", "") for candidate in route.candidates]
                suffix = f" 候选：{_join(names, language='zh')}。" if names else ""
                return "这个名称对应多个目录实体，请提供更具体的 ID 或内部名称。" + suffix
            return route.clarification_question
        return "需要更多信息才能确定实体。" if language == "zh" else "More information is required to identify the entity."

    def _not_found(self, route: RouteDecision, language: str) -> str:
        if route.candidates:
            names = [candidate.get("name", "") for candidate in route.candidates]
            if language == "zh":
                return f"没有找到精确匹配。可能的候选是：{_join(names, language='zh')}。"
            return f"No exact match was found. Possible matches: {_join(names, language='en')}."
        entity = route.entity or "该查询"
        if language == "zh":
            return f"当前 Terraria 目录没有找到“{entity}”的可用结构化证据，我不会据此编造答案。"
        return f"The Terraria catalog contains no usable structured evidence for {entity!r}, so I will not invent an answer."

    def _render_item(self, facts: dict[str, Any], language: str) -> str:
        name = facts.get("name")
        item_id = facts.get("item_id")
        classification = facts.get("classification") or {}
        combat = facts.get("combat") or {}
        inventory = facts.get("inventory") or {}
        economy = facts.get("economy") or {}
        parts = [f"{name}（物品 ID {item_id}）" if language == "zh" else f"{name} (item ID {item_id})"]

        types = classification.get("types") or []
        if types:
            parts.append(("类型：" if language == "zh" else "Type: ") + _join(types, language=language))

        attributes = []
        damage_type = combat.get("damage_type")
        if damage_type:
            attributes.append(
                f"伤害类型 {damage_type}" if language == "zh" else f"damage type {damage_type}"
            )
        for key, label_en, label_zh in (
            ("damage", "damage", "伤害"),
            ("defense", "defense", "防御"),
            ("critical_chance_percent", "critical chance", "暴击率"),
            ("knockback", "knockback", "击退"),
            ("mana_cost", "mana", "魔力消耗"),
            ("use_time", "use time", "使用时间"),
        ):
            display = _number_display(combat.get(key))
            if display is not None:
                suffix = "%" if key == "critical_chance_percent" else ""
                attributes.append(
                    f"{label_zh if language == 'zh' else label_en} {display}{suffix}"
                )
        if attributes:
            parts.append(("战斗属性：" if language == "zh" else "Combat: ") + _join(attributes, language=language))

        stack = _number_display((inventory.get("stack_limit") or {}).get("primary"))
        rarity = _number_display((inventory.get("rarity") or {}).get("primary"))
        inventory_parts = []
        if stack is not None:
            inventory_parts.append(("堆叠上限 " if language == "zh" else "stack limit ") + stack)
        if rarity is not None:
            inventory_parts.append(("稀有度 " if language == "zh" else "rarity ") + rarity)
        if inventory_parts:
            parts.append(("物品栏：" if language == "zh" else "Inventory: ") + _join(inventory_parts, language=language))

        sell = ((economy.get("sell") or {}).get("primary_copper"))
        sell_display = _coin_display(sell, language=language)
        if sell_display:
            parts.append(("售价：" if language == "zh" else "Sell value: ") + sell_display)

        tooltip = facts.get("tooltip")
        if tooltip:
            parts.append(("说明：" if language == "zh" else "Tooltip: ") + str(tooltip))
        return "\n".join(parts)

    def _render_npc(self, facts: dict[str, Any], mode: str, language: str) -> str:
        name = facts.get("name")
        npc_id = facts.get("npc_id")
        parts = [f"{name}（NPC ID {npc_id}）" if language == "zh" else f"{name} (NPC ID {npc_id})"]
        types = facts.get("npc_types") or []
        if types:
            parts.append(("类型：" if language == "zh" else "Type: ") + _join(types, language=language))
        environment = facts.get("environment") or []
        if environment:
            parts.append(("环境：" if language == "zh" else "Environment: ") + _join(environment, language=language))

        stats = facts.get("stats") or {}
        stat_parts = []
        for key, en, zh in (("life", "life", "生命"), ("damage", "damage", "伤害"), ("defense", "defense", "防御")):
            display = _number_display(_mode_value(stats.get(key), mode))
            if display is not None:
                stat_parts.append(f"{zh if language == 'zh' else en} {display}")
        if stat_parts:
            mode_label = {"normal": "普通", "expert": "专家", "master": "大师"}.get(mode, mode) if language == "zh" else mode
            parts.append((f"{mode_label}模式属性：" if language == "zh" else f"{mode.title()} stats: ") + _join(stat_parts, language=language))

        immunities = facts.get("immunities") or []
        if immunities:
            parts.append(("免疫：" if language == "zh" else "Immunities: ") + _join(immunities, language=language))
        return "\n".join(parts)

    def _render_recipe(self, facts: dict[str, Any], language: str) -> str:
        result_name = facts.get("result_name")
        variants = list(facts.get("variants") or [])
        header = (
            f"{result_name} 有 {len(variants)} 个当前可用配方方案："
            if language == "zh"
            else f"{result_name} has {len(variants)} selected recipe variant(s):"
        )
        lines = [header]
        for index, variant in enumerate(variants[: self.max_entries], start=1):
            ingredients = [f"{item['name']} ×{item['quantity']}" for item in variant.get("ingredients") or []]
            stations = variant.get("stations") or []
            result_quantity = variant.get("result_quantity")
            label = variant.get("version_label") or (f"方案 {index}" if language == "zh" else f"Option {index}")
            lines.append(f"{label}：" if language == "zh" else f"{label}:")
            lines.append(("  材料：" if language == "zh" else "  Ingredients: ") + _join(ingredients, language=language))
            lines.append(("  制作站：" if language == "zh" else "  Station: ") + _join(stations, language=language))
            if result_quantity not in {None, 1}:
                lines.append(("  产出数量：" if language == "zh" else "  Output quantity: ") + str(result_quantity))
        if len(variants) > self.max_entries:
            lines.append((f"仅显示前 {self.max_entries} 个方案。" if language == "zh" else f"Only the first {self.max_entries} variants are shown."))
        return "\n".join(lines)

    def _render_reverse_recipe(self, facts: dict[str, Any], language: str) -> str:
        item = facts.get("item") or {}
        recipes = list(facts.get("recipes") or [])
        name = item.get("name")
        if not recipes:
            return f"没有找到使用 {name} 的首选配方。" if language == "zh" else f"No preferred recipes using {name} were found."
        names = [recipe.get("result_name") for recipe in recipes[: self.max_entries]]
        prefix = f"{name} 可以用于合成：" if language == "zh" else f"{name} is used in these recipes: "
        suffix = "、".join(names) if language == "zh" else ", ".join(names)
        if len(recipes) > self.max_entries:
            suffix += f"（仅显示前 {self.max_entries} 个）" if language == "zh" else f" (first {self.max_entries} shown)"
        return prefix + suffix + "。"

    def _render_drops(self, facts: dict[str, Any], intent: AssistantIntent, language: str) -> str:
        drops = list(facts.get("drops") or [])
        mode = facts.get("mode", "normal")
        mode_display = (
            {"normal": "普通", "expert": "专家", "master": "大师"}.get(mode, mode)
            if language == "zh"
            else mode
        )
        if not drops:
            return "该模式下没有匹配的掉落记录。" if language == "zh" else "No matching drop records are available for that mode."
        if intent == AssistantIntent.DROPS_FOR_ITEM:
            subject = drops[0].get("item_name")
            header = f"{subject} 的掉落来源（{mode_display}）：" if language == "zh" else f"Sources for {subject} ({mode_display}):"
        else:
            subject = drops[0].get("source_name")
            header = f"{subject} 的掉落物（{mode_display}）：" if language == "zh" else f"Drops from {subject} ({mode_display}):"
        lines = [header]
        for drop in drops[: self.max_entries]:
            chance = (drop.get("chance") or {}).get("display") or "unknown"
            quantity = (drop.get("quantity") or {}).get("display") or "unknown"
            if intent == AssistantIntent.DROPS_FOR_ITEM:
                label = drop.get("source_name")
            else:
                label = drop.get("item_name")
            condition = drop.get("conditions") or []
            line = f"- {label}: {chance}, ×{quantity}"
            if condition:
                line += ("；条件：" if language == "zh" else "; conditions: ") + _join(condition, language=language)
            lines.append(line)
        if len(drops) > self.max_entries:
            lines.append((f"仅显示前 {self.max_entries} 条。" if language == "zh" else f"Only the first {self.max_entries} records are shown."))
        return "\n".join(lines)

    def _render_search(self, facts: dict[str, Any], language: str) -> str:
        items = list(facts.get("items") or [])
        npcs = list(facts.get("npcs") or [])
        recipes = list(facts.get("recipes") or [])
        if not any((items, npcs, recipes)):
            return "目录中没有找到匹配项。" if language == "zh" else "No matching catalog entries were found."
        lines = ["搜索结果：" if language == "zh" else "Catalog matches:"]
        if items:
            lines.append(("- 物品：" if language == "zh" else "- Items: ") + _join([row["name"] for row in items[: self.max_entries]], language=language))
        if npcs:
            lines.append(("- NPC：" if language == "zh" else "- NPCs: ") + _join([row["name"] for row in npcs[: self.max_entries]], language=language))
        if recipes:
            lines.append(("- 配方结果：" if language == "zh" else "- Recipe results: ") + _join([row["result_name"] for row in recipes[: self.max_entries]], language=language))
        return "\n".join(lines)

    def render(
        self,
        request: AssistantRequest,
        route: RouteDecision,
        retrieval: dict[str, Any],
        *,
        language: str,
    ) -> str:
        if route.needs_clarification:
            return self._clarification(route, language)

        status = retrieval.get("status")
        if status in {"ambiguous", "family"}:
            return self._clarification(route, language)
        if status == "not_found":
            return self._not_found(route, language)

        facts = retrieval.get("facts") or {}
        if route.intent == AssistantIntent.ITEM:
            return self._render_item(facts, language)
        if route.intent == AssistantIntent.NPC:
            return self._render_npc(facts, str(route.parameters.get("mode", request.mode)), language)
        if route.intent == AssistantIntent.RECIPE:
            return self._render_recipe(facts, language)
        if route.intent == AssistantIntent.RECIPES_USING_ITEM:
            return self._render_reverse_recipe(facts, language)
        if route.intent in {AssistantIntent.DROPS_FOR_ITEM, AssistantIntent.DROPS_FROM_SOURCE}:
            return self._render_drops(facts, route.intent, language)
        return self._render_search(facts, language)
