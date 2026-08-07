"""Deterministic evidence renderer for Stardew Valley."""

from __future__ import annotations

from typing import Any


def _citation(evidence: list[dict[str, Any]]) -> str:
    return " " + " ".join(f"[{item['source_id']}]" for item in evidence[:3]) if evidence else ""


_ZH_TERMS = {
    "spring": "春季",
    "summer": "夏季",
    "fall": "秋季",
    "winter": "冬季",
    "any": "任意天气",
    "rain": "雨天",
    "rain_totem": "使用雨水图腾的雨天",
    "sunny": "晴天",
    "storm": "雷雨天",
    "snow": "雪天",
    "river": "河流",
    "secret_woods": "秘密森林",
    "witchs_swamp": "女巫沼泽",
    "mountain_lake": "山地湖泊",
    "forest_pond": "森林池塘",
    "ginger_island_freshwater": "姜岛淡水区域",
    "ginger_island_ocean": "姜岛海域",
    "ocean": "海洋",
    "forest_river": "森林河流",
    "forest_pond": "森林池塘",
    "desert_pond": "沙漠池塘",
    "sewers": "下水道",
    "pirate_cove": "海盗湾",
    "night_market_submarine": "夜市潜水艇",
    "mines_20": "矿井20层",
    "mines_60": "矿井60层",
    "mines_100": "矿井100层",
    "volcano_caldera": "火山口",
    "Wood": "木材",
    "Copper Bar": "铜锭",
    "Iron Bar": "铁锭",
    "Oak Resin": "橡树树脂",
    "Coal": "煤炭",
    "Stone": "石头",
    "Lake Fish Bundle": "湖鱼收集包",
    "River Fish Bundle": "河鱼收集包",
    "Night Fishing Bundle": "夜间钓鱼收集包",
    "Spring Foraging Bundle": "春季采集收集包",
    "Fish Tank": "鱼缸",
    "Crafts Room": "工艺室",
    "farming": "耕种",
}


def _zh(value: object) -> str:
    text = str(value)
    return _ZH_TERMS.get(text, _ZH_TERMS.get(text.casefold(), text))


def _display_time(value: object, *, zh: bool) -> str:
    text = str(value)
    mapping_en = {
        "00:00": "12:00 AM",
        "02:00": "2:00 AM",
        "06:00": "6:00 AM",
        "12:00": "12:00 PM",
        "16:00": "4:00 PM",
        "18:00": "6:00 PM",
        "19:00": "7:00 PM",
        "22:00": "10:00 PM",
        "24:00": "12:00 AM",
        "26:00": "2:00 AM",
    }
    mapping_zh = {
        "00:00": "午夜 12:00",
        "02:00": "凌晨 2:00",
        "06:00": "早上 6:00",
        "12:00": "中午 12:00",
        "16:00": "下午 4:00",
        "18:00": "下午 6:00",
        "19:00": "晚上 7:00",
        "22:00": "晚上 10:00",
        "24:00": "午夜 12:00",
        "26:00": "次日凌晨 2:00",
    }
    return (mapping_zh if zh else mapping_en).get(text, text)


class StardewRenderer:
    def render(
        self,
        retrieval: dict[str, Any],
        *,
        language: str,
        evidence: list[dict[str, Any]],
    ) -> str:
        status = retrieval.get("status")
        intent = retrieval.get("intent")
        entity = retrieval.get("entity")
        facts = retrieval.get("facts") or {}
        cite = _citation(evidence)
        zh = language == "zh"

        if status == "needs_context":
            missing = ", ".join(retrieval.get("missing_context") or [])
            return (
                f"我还需要这些玩家状态才能可靠回答：{missing}。"
                if zh else f"I need the following player-state fields before I can answer reliably: {missing}."
            )
        if status == "ambiguous":
            names = ", ".join(item.get("name", "") for item in retrieval.get("candidates") or [])
            return (f"这个名称对应多个实体：{names}。请说明你指的是哪一个。" if zh else f"This name matches multiple entities: {names}. Please clarify which one you mean.")
        if status == "partial":
            warning = " ".join(retrieval.get("warnings") or [])
            if zh:
                return f"当前快照只能提供部分信息，不能把标准收集包当作混合收集包回答。{warning}{cite}"
            return f"The current snapshot can only provide a partial answer and will not present Standard Bundle data as Remixed Bundle data. {warning}{cite}"
        if status != "found":
            return (
                f"本地星露谷知识库中没有足够证据支持关于“{entity or retrieval.get('query')}”的答案，所以我不会猜测。"
                if zh else f"The local Stardew Valley knowledge base has no usable evidence for '{entity or retrieval.get('query')}', so I will not guess."
            )

        if intent == "crop_info":
            seasons = ", ".join(_zh(value) if zh else str(value).title() for value in (facts.get("seasons") or []))
            regrow = facts.get("regrow_days")
            location_seasons = facts.get("location_seasons") or {}
            exception_parts = []
            for location, season_values in location_seasons.items():
                season_values = list(season_values or [])
                all_seasons = set(season_values) == {"spring", "summer", "fall", "winter"}
                if zh:
                    location_label = {
                        "valley": "星露谷本土",
                        "ginger_island": "姜岛",
                        "indoors": "室内",
                    }.get(str(location), _zh(location))
                    season_label = "全年" if all_seasons else "/".join(_zh(value) for value in season_values)
                    exception_parts.append(f"{location_label}：{season_label}")
                else:
                    location_label = {
                        "valley": "the Valley",
                        "ginger_island": "Ginger Island",
                        "indoors": "indoors",
                    }.get(str(location), str(location).replace("_", " ").title())
                    season_label = "all seasons" if all_seasons else "/".join(str(value).title() for value in season_values)
                    exception_parts.append(f"{location_label}: {season_label}")
            if zh:
                answer = f"{entity} 在星露谷本土可于 {seasons} 种植，成熟需要 {facts.get('growth_days')} 天"
                answer += f"，首次收获后每 {regrow} 天再生" if regrow else "，收获后不会再生"
                if exception_parts:
                    answer += "。地点规则：" + "；".join(exception_parts)
                answer += f"。普通品质基础售价为 {facts.get('base_sell_price')}g。{cite}"
                return answer
            answer = f"{entity} grows in {seasons} in the Valley and takes {facts.get('growth_days')} days to mature"
            answer += f", then regrows every {regrow} days" if regrow else " and does not regrow"
            if exception_parts:
                answer += ". Location rules: " + "; ".join(exception_parts)
            return answer + f". Its base sell price is {facts.get('base_sell_price')}g.{cite}"

        if intent == "crop_deadline":
            if zh:
                verdict = "来得及" if facts.get("can_harvest_before_season_end") else "来不及"
                harvest_detail = (
                    f"首次收获日为{_zh(facts.get('season'))}第 {facts.get('first_harvest_day')} 天。"
                    if facts.get("first_harvest_day") is not None
                    else "该季节内无法成熟。"
                )
                return (
                    f"{entity} 成熟需要 {facts.get('growth_days')} 天。如果在 "
                    f"{_zh(facts.get('season'))}第 {facts.get('planting_day')} 天种下，"
                    f"在不使用生长加速的情况下{verdict}在季末前收获。{harvest_detail}"
                    f"最晚种植日是第 {facts.get('latest_planting_day')} 天；预计可收获 "
                    f"{facts.get('estimated_harvests_before_season_end')} 次。{cite}"
                )
            verdict = "can" if facts.get("can_harvest_before_season_end") else "cannot"
            harvest_detail = (
                f"The first harvest is on {str(facts.get('season')).title()} "
                f"{facts.get('first_harvest_day')}. "
                if facts.get("first_harvest_day") is not None
                else "It will not mature during that season. "
            )
            return (
                f"{entity} takes {facts.get('growth_days')} days to mature. If planted on "
                f"{str(facts.get('season')).title()} day {facts.get('planting_day')}, it {verdict} "
                f"be harvested before the season ends without growth modifiers. {harvest_detail}"
                f"The latest planting day is day {facts.get('latest_planting_day')}; estimated harvests: "
                f"{facts.get('estimated_harvests_before_season_end')}.{cite}"
            )

        if intent == "fish_availability":
            windows = facts.get("matching_windows") or facts.get("availability_windows") or []
            lines = []
            for window in windows:
                seasons_raw = list(window.get("seasons") or [])
                weather_raw = list(window.get("weather") or [])
                locations_raw = list(window.get("locations") or [])
                all_seasons = set(seasons_raw) == {"spring", "summer", "fall", "winter"}
                if zh:
                    seasons = "全年" if all_seasons else "/".join(_zh(value) for value in seasons_raw)
                    weather = "/".join(_zh(value) for value in weather_raw)
                    locations = "、".join(_zh(value) for value in locations_raw)
                else:
                    seasons = "all seasons" if all_seasons else "/".join(str(value).title() for value in seasons_raw)
                    weather = "/".join(str(value).replace("_", " ").title() for value in weather_raw)
                    locations = ", ".join(str(value).replace("_", " ").title() for value in locations_raw)
                start = _display_time(window.get("time_start"), zh=zh)
                end = _display_time(window.get("time_end"), zh=zh)
                if window.get("time_start") == "00:00" and window.get("time_end") == "24:00":
                    time_text = "任意时间" if zh else "any time"
                else:
                    time_text = f"{start}–{end}"
                lines.append(f"{seasons}; {weather}; {time_text}; {locations}")
            if zh:
                return f"{entity} 的可钓条件：\n- " + "\n- ".join(lines) + cite
            return f"{entity} can be caught under these conditions:\n- " + "\n- ".join(lines) + cite

        if intent == "villager_gifts":
            gifts = ", ".join(facts.get("loved_gifts") or [])
            birthday = facts.get("birthday") or {}
            if zh:
                return f"{entity} 最喜欢的礼物包括：{gifts}。生日是{_zh(birthday.get('season'))}第 {birthday.get('day')} 天。{cite}"
            return f"{entity}'s loved gifts include: {gifts}. Birthday: {str(birthday.get('season')).title()} {birthday.get('day')}.{cite}"

        if intent == "villager_info":
            birthday = facts.get("birthday") or {}
            if zh:
                return f"{entity} 住在 {facts.get('home')}，生日是{_zh(birthday.get('season'))}第 {birthday.get('day')} 天，可结婚：{'是' if facts.get('marriageable') else '否'}。{cite}"
            return f"{entity} lives at {facts.get('home')}. Birthday: {str(birthday.get('season')).title()} {birthday.get('day')}. Marriageable: {facts.get('marriageable')}.{cite}"

        if intent == "recipe":
            ingredients = ", ".join(
                f"{_zh(item['item_name']) if zh else item['item_name']} ×{item['quantity']}"
                for item in facts.get("ingredients") or []
            )
            unlock = facts.get("unlock") or {}
            source = str(unlock.get("source") or facts.get("unlock_source") or "unknown")
            if unlock.get("skill") is not None and unlock.get("level") is not None:
                unlock_text_en = f"{str(unlock.get('skill')).title()} level {unlock.get('level')}"
                unlock_text_zh = f"{_zh(unlock.get('skill'))} {unlock.get('level')} 级"
            else:
                unlock_text_en = source
                unlock_text_zh = source
            action_en = "Cook" if facts.get("recipe_type") == "cooking" else "Craft"
            action_zh = "烹饪" if facts.get("recipe_type") == "cooking" else "制作"
            if zh:
                return f"{action_zh} {entity} 需要：{ingredients}。解锁条件：{unlock_text_zh}。{cite}"
            return f"{action_en} {entity} with: {ingredients}. Unlock: {unlock_text_en}.{cite}"

        if intent == "recipes_using_item":
            recipes = ", ".join(facts.get("recipes") or [])
            return (f"{entity} 可用于这些已收录配方：{recipes}。{cite}" if zh else f"Tracked recipes using {entity}: {recipes}.{cite}")

        if intent == "bundle":
            requirements = ", ".join(
                f"{item['item_name']} ×{item.get('quantity', 1)}" +
                (f" ({item.get('minimum_quality')})" if item.get('minimum_quality') else "")
                for item in facts.get("requirements") or []
            )
            selection = facts.get("selection_rule") or "all"
            if zh:
                return f"{entity} 位于 {_zh(facts.get('room'))}，要求：{requirements}。选择规则：{selection}；模式：{facts.get('bundle_mode')}。{cite}"
            return f"{entity} is in the {facts.get('room')} and requires: {requirements}. Selection rule: {selection}; mode: {facts.get('bundle_mode')}.{cite}"

        if intent == "bundles_requiring_item":
            bundles = ", ".join(_zh(value) if zh else str(value) for value in (facts.get("bundles") or []))
            return (f"{entity} 被这些已收录收集包需要：{bundles}。{cite}" if zh else f"Tracked bundles requiring {entity}: {bundles}.{cite}")

        if intent == "acquisition":
            sources = facts.get("sources") or []
            lines = []
            for source in sources[:8]:
                source_name = source.get("source_name")
                source_type = source.get("source_type")
                location = source.get("location")
                price = source.get("price")
                currency = source.get("currency") or "g"
                conditions = "; ".join(str(item) for item in source.get("conditions") or [])
                if zh:
                    detail = f"{source_name}（{source_type}）"
                    if location:
                        detail += f"，地点：{_zh(location)}"
                    if price is not None:
                        detail += f"，价格：{price}{currency}"
                    if conditions:
                        detail += f"，条件：{conditions}"
                else:
                    detail = f"{source_name} ({source_type})"
                    if location:
                        detail += f", location: {str(location).replace('_', ' ')}"
                    if price is not None:
                        detail += f", price: {price} {currency}"
                    if conditions:
                        detail += f", conditions: {conditions}"
                lines.append(detail)
            intro = f"{entity} 的获取方式：" if zh else f"Tracked ways to obtain {entity}:"
            return intro + "\n- " + "\n- ".join(lines) + cite

        if intent == "guide":
            hits = facts.get("hits") or []
            if not hits:
                return "没有检索到足够相关的星露谷攻略证据。" if zh else "No sufficiently relevant Stardew Valley guide evidence was retrieved."
            lines = []
            for hit in hits[:4]:
                summary = str(hit.get("text", "")).strip().replace("\n", " ")
                if len(summary) > 500:
                    summary = summary[:497].rstrip() + "..."
                lines.append(f"- {hit.get('citation_label')}: {summary} [{hit.get('source_id')}]")
            intro = "从本地星露谷攻略库检索到：" if zh else "Relevant local Stardew Valley guide evidence:"
            return intro + "\n" + "\n".join(lines)

        if intent == "entity":
            return (f"{entity}：{facts}。{cite}" if zh else f"{entity}: {facts}.{cite}")
        if intent == "search":
            names = ", ".join(item.get("name", "") for item in facts.get("matches") or [])
            return (f"找到这些相关实体：{names}。{cite}" if zh else f"Related entities: {names}.{cite}")
        return (f"已找到 {entity} 的证据。{cite}" if zh else f"Evidence found for {entity}.{cite}")
