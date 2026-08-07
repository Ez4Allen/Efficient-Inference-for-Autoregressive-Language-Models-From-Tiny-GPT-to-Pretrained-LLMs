#!/usr/bin/env python3
"""Generate the curated, versioned Stardew Valley release snapshot.

The generated data is intentionally deterministic and source-attributed. It is
large enough to satisfy the course MVP contract while remaining compact enough
to inspect and ship in Git. The script writes structured facts, acquisition
relations, the offline guide seed, a machine-validated benchmark, and a small
training-only grounded QA set. It never labels AI-generated content as human
reviewed.
"""

from __future__ import annotations

import hashlib
import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.games.stardew.normalizers import normalize_name
from src.utils.io import write_json, write_jsonl

GAME_VERSION = "1.6.15"
RETRIEVED_AT = "2026-08-05T00:00:00+00:00"
WIKI = "https://stardewvalleywiki.com"
LICENSE_NAME = "CC BY-NC-SA 3.0"


def slug(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value.casefold()).strip("-")


def wiki_url(page_title: str) -> str:
    return f"{WIKI}/{quote(page_title.replace(' ', '_'), safe='()_\'') }"


def provenance(page_title: str, section_title: str) -> dict[str, Any]:
    return {
        "source_name": "Official Stardew Valley Wiki",
        "page_title": page_title,
        "section_title": section_title,
        "source_url": wiki_url(page_title),
        "revision_id": None,
        "retrieved_at": RETRIEVED_AT,
        "license_name": LICENSE_NAME,
    }


def base_record(
    record_type: str,
    name: str,
    *,
    aliases: Iterable[str] = (),
    facts: dict[str, Any],
    conditions: dict[str, Any] | None = None,
    page_title: str | None = None,
    section_title: str = "Information",
    parse_status: str = "ok",
    parse_warnings: Iterable[str] = (),
    record_id_suffix: str | None = None,
) -> dict[str, Any]:
    suffix = record_id_suffix or slug(name)
    return {
        "schema_version": 1,
        "game": "stardew_valley",
        "game_version": GAME_VERSION,
        "platform": "all",
        "record_type": record_type,
        "source_catalog_id": f"stardew:{record_type}:{suffix}",
        "name": name,
        "normalized_name": normalize_name(name),
        "aliases": list(dict.fromkeys(str(item).strip() for item in aliases if str(item).strip())),
        "facts": facts,
        "conditions": conditions or {},
        "provenance": provenance(page_title or name, section_title),
        "parse_status": parse_status,
        "parse_warnings": list(parse_warnings),
    }


def crop(
    name: str,
    seasons: list[str],
    growth: int,
    sell: int,
    *,
    aliases: Iterable[str] = (),
    regrow: int | None = None,
    minimum: int = 1,
    trellis: bool = False,
    giant: bool = False,
    seed_prices: dict[str, int] | None = None,
    seed_name: str | None = None,
    notes: Iterable[str] = (),
    locations_or_exceptions: Iterable[str] = (),
    location_seasons: dict[str, Iterable[str]] | None = None,
) -> dict[str, Any]:
    facts = {
        "seed_name": seed_name or f"{name} Seeds",
        "seasons": seasons,
        "growth_days": growth,
        "regrow_days": regrow,
        "harvest_quantity": {"minimum": minimum, "maximum": None if minimum > 1 or regrow else 1},
        "trellis": trellis,
        "giant_crop": giant,
        "seed_buy_prices": seed_prices or {},
        "base_sell_price": sell,
        "locations_or_exceptions": list(locations_or_exceptions),
        "location_seasons": {
            str(location): list(season_values)
            for location, season_values in (location_seasons or {}).items()
        },
    }
    if notes:
        facts["notes"] = list(notes)
    return base_record(
        "crop",
        name,
        aliases=aliases,
        facts=facts,
        conditions={"seasons": seasons},
        page_title=name,
        section_title="Information",
    )


CROPS = [
    crop("Parsnip", ["spring"], 4, 35, aliases=["欧洲防风草"], seed_prices={"pierre": 20, "joja": 25}),
    crop("Potato", ["spring"], 6, 80, aliases=["土豆", "马铃薯"], seed_prices={"pierre": 50, "joja": 62}, notes=["Each harvest can yield extra Potatoes."]),
    crop("Cauliflower", ["spring"], 12, 175, aliases=["花椰菜", "菜花"], giant=True, seed_prices={"pierre": 80, "joja": 100}),
    crop("Green Bean", ["spring"], 10, 40, aliases=["青豆"], regrow=3, trellis=True, seed_prices={"pierre": 60, "joja": 75}),
    crop("Kale", ["spring"], 6, 110, aliases=["甘蓝"], seed_prices={"pierre": 70, "joja": 87}),
    crop("Strawberry", ["spring"], 8, 120, aliases=["草莓"], regrow=4, seed_prices={"egg_festival": 100}),
    crop("Garlic", ["spring"], 4, 60, aliases=["大蒜"], seed_prices={"pierre": 40}),
    crop("Rhubarb", ["spring"], 13, 220, aliases=["大黄"], seed_prices={"oasis": 100}),
    crop("Coffee Bean", ["spring", "summer"], 10, 15, aliases=["咖啡豆"], regrow=2, minimum=4, seed_name="Coffee Bean", notes=["Can be planted in Spring or Summer."]),
    crop("Blue Jazz", ["spring"], 7, 50, aliases=["蓝爵士"], seed_prices={"pierre": 30, "joja": 37}),
    crop("Tulip", ["spring"], 6, 30, aliases=["郁金香"], seed_prices={"pierre": 20, "joja": 25}),
    crop("Unmilled Rice", ["spring"], 8, 30, aliases=["未碾米"], seed_name="Rice Shoot", seed_prices={"pierre": 40}, notes=["Grows faster near water."]),
    crop("Blueberry", ["summer"], 13, 50, aliases=["蓝莓"], regrow=4, minimum=3, seed_prices={"pierre": 80}),
    crop("Melon", ["summer"], 12, 250, aliases=["甜瓜"], giant=True, seed_prices={"pierre": 80, "joja": 100}),
    crop("Tomato", ["summer"], 11, 60, aliases=["番茄", "西红柿"], regrow=4, seed_prices={"pierre": 50, "joja": 62}),
    crop("Hot Pepper", ["summer"], 5, 40, aliases=["辣椒"], regrow=3, seed_prices={"pierre": 40, "joja": 50}),
    crop("Wheat", ["summer", "fall"], 4, 25, aliases=["小麦"], seed_prices={"pierre": 10, "joja": 12}),
    crop("Summer Spangle", ["summer"], 8, 90, aliases=["夏季亮片"], seed_prices={"pierre": 50, "joja": 62}),
    crop("Poppy", ["summer"], 7, 140, aliases=["虞美人"], seed_prices={"pierre": 100, "joja": 125}),
    crop("Radish", ["summer"], 6, 90, aliases=["萝卜"], seed_prices={"pierre": 40, "joja": 50}),
    crop("Red Cabbage", ["summer"], 9, 260, aliases=["红叶卷心菜"], seed_prices={"pierre": 100}),
    crop("Starfruit", ["summer"], 13, 750, aliases=["杨桃"], seed_prices={"oasis": 400}),
    crop("Hops", ["summer"], 11, 25, aliases=["啤酒花"], regrow=1, trellis=True, seed_prices={"pierre": 60, "joja": 75}),
    crop("Corn", ["summer", "fall"], 14, 50, aliases=["玉米"], regrow=4, seed_prices={"pierre": 150, "joja": 187}),
    crop("Sunflower", ["summer", "fall"], 8, 80, aliases=["向日葵"], seed_prices={"pierre": 200, "joja": 125}),
    crop("Cranberries", ["fall"], 7, 75, aliases=["蔓越莓"], regrow=5, minimum=2, seed_prices={"pierre": 240, "joja": 300}),
    crop("Pumpkin", ["fall"], 13, 320, aliases=["南瓜"], giant=True, seed_prices={"pierre": 100, "joja": 125}),
    crop("Yam", ["fall"], 10, 160, aliases=["山药"], seed_prices={"pierre": 60, "joja": 75}),
    crop("Bok Choy", ["fall"], 4, 80, aliases=["小白菜"], seed_prices={"pierre": 50, "joja": 62}),
    crop("Eggplant", ["fall"], 5, 60, aliases=["茄子"], regrow=5, seed_prices={"pierre": 20, "joja": 25}),
    crop("Grape", ["fall"], 10, 80, aliases=["葡萄"], regrow=3, trellis=True, seed_prices={"pierre": 60, "joja": 75}),
    crop("Amaranth", ["fall"], 7, 150, aliases=["苋菜"], seed_prices={"pierre": 70, "joja": 87}),
    crop("Artichoke", ["fall"], 8, 160, aliases=["朝鲜蓟"], seed_prices={"pierre": 30}),
    crop("Beet", ["fall"], 6, 100, aliases=["甜菜"], seed_prices={"oasis": 20}),
    crop("Fairy Rose", ["fall"], 12, 290, aliases=["仙女玫瑰"], seed_prices={"pierre": 200, "joja": 250}),
    crop("Sweet Gem Berry", ["fall"], 24, 3000, aliases=["宝石甜莓"], seed_name="Rare Seed", seed_prices={"traveling_cart": 1000}),
    crop("Powdermelon", ["winter"], 7, 60, aliases=["粉末瓜"], giant=True, seed_prices={}, notes=["A winter crop introduced in version 1.6."]),
    crop(
        "Pineapple",
        ["summer"],
        14,
        300,
        aliases=["菠萝"],
        regrow=7,
        seed_prices={},
        notes=["Grows in Summer in the Valley and in all seasons on Ginger Island."],
        locations_or_exceptions=["all seasons on Ginger Island"],
        location_seasons={
            "valley": ["summer"],
            "ginger_island": ["spring", "summer", "fall", "winter"],
        },
    ),
    crop(
        "Taro Root",
        ["summer"],
        10,
        100,
        aliases=["芋头"],
        seed_name="Taro Tuber",
        notes=[
            "Grows in Summer in the Valley and in all seasons on Ginger Island.",
            "Matures in 7 days and does not need watering when planted near water.",
        ],
        locations_or_exceptions=["all seasons on Ginger Island", "7-day growth near water"],
        location_seasons={
            "valley": ["summer"],
            "ginger_island": ["spring", "summer", "fall", "winter"],
        },
    ),
    crop("Ancient Fruit", ["spring", "summer", "fall"], 28, 550, aliases=["上古水果"], regrow=7, seed_name="Ancient Seeds"),
    crop(
        "Cactus Fruit",
        ["spring", "summer", "fall", "winter"],
        12,
        75,
        aliases=["仙人掌果子"],
        regrow=3,
        seed_name="Cactus Seeds",
        notes=["Must be grown indoors, in the Greenhouse, in Garden Pots, or on Ginger Island."],
        locations_or_exceptions=["indoors or on Ginger Island only"],
        location_seasons={
            "indoors": ["spring", "summer", "fall", "winter"],
            "ginger_island": ["spring", "summer", "fall", "winter"],
        },
    ),
]


def fish_window(
    seasons: Iterable[str],
    weather: Iterable[str],
    start: str,
    end: str,
    locations: Iterable[str],
    **extra: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "seasons": list(seasons),
        "weather": list(weather),
        "time_start": start,
        "time_end": end,
        "locations": list(locations),
    }
    result.update(extra)
    return result


def fish(
    name: str,
    windows: list[dict[str, Any]],
    *,
    aliases: Iterable[str] = (),
    difficulty: int | None = None,
    behavior: str | None = None,
    special_conditions: Iterable[str] = (),
) -> dict[str, Any]:
    facts = {
        "difficulty": difficulty,
        "behavior": behavior,
        "base_sell_prices": {},
        "availability_windows": windows,
        "special_conditions": list(special_conditions),
    }
    warnings = [] if difficulty is not None and behavior else ["Difficulty or movement behavior is not included in this compact snapshot."]
    return base_record(
        "fish",
        name,
        aliases=aliases,
        facts=facts,
        page_title=name,
        section_title="Information",
        parse_status="ok" if not warnings else "partial",
        parse_warnings=warnings,
    )


ALL_SEASONS = ["spring", "summer", "fall", "winter"]
ANY = ["any"]
FISH = [
    fish("Anchovy", [fish_window(["spring", "fall"], ANY, "06:00", "26:00", ["ocean"])], aliases=["凤尾鱼"]),
    fish("Sardine", [fish_window(["spring", "fall", "winter"], ANY, "06:00", "19:00", ["ocean"])], aliases=["沙丁鱼"]),
    fish("Tuna", [fish_window(["summer", "winter"], ANY, "06:00", "19:00", ["ocean"])], aliases=["金枪鱼"]),
    fish("Red Snapper", [fish_window(["summer", "fall"], ["rain"], "06:00", "19:00", ["ocean"]), fish_window(["winter"], ["rain_totem"], "06:00", "19:00", ["ocean"])], aliases=["红鲷鱼"]),
    fish("Tilapia", [fish_window(["summer", "fall"], ANY, "06:00", "14:00", ["ocean"])], aliases=["罗非鱼"]),
    fish("Halibut", [fish_window(["spring", "summer", "winter"], ANY, "06:00", "11:00", ["ocean"]), fish_window(["spring", "summer", "winter"], ANY, "19:00", "26:00", ["ocean"])], aliases=["大比目鱼"]),
    fish("Herring", [fish_window(["spring", "winter"], ANY, "06:00", "26:00", ["ocean"])], aliases=["鲱鱼"]),
    fish("Eel", [fish_window(["spring", "fall"], ["rain"], "16:00", "26:00", ["ocean"])], aliases=["鳗鱼"], difficulty=70, behavior="smooth"),
    fish("Octopus", [fish_window(["summer"], ANY, "06:00", "13:00", ["ocean"])], aliases=["章鱼"]),
    fish("Pufferfish", [fish_window(["summer"], ["sunny"], "12:00", "16:00", ["ocean"])], aliases=["河豚"]),
    fish("Super Cucumber", [fish_window(["summer", "fall"], ANY, "18:00", "26:00", ["ocean"])], aliases=["大海参"]),
    fish("Flounder", [fish_window(["spring", "summer"], ANY, "06:00", "20:00", ["ocean"])], aliases=["比目鱼"]),
    fish("Red Mullet", [fish_window(["summer", "winter"], ANY, "06:00", "19:00", ["ocean"])], aliases=["红鲻鱼"]),
    fish("Squid", [fish_window(["winter"], ANY, "18:00", "26:00", ["ocean"])], aliases=["鱿鱼"]),
    fish("Sea Cucumber", [fish_window(["fall", "winter"], ANY, "06:00", "19:00", ["ocean"])], aliases=["海参"]),
    fish("Albacore", [fish_window(["fall", "winter"], ANY, "06:00", "11:00", ["ocean"]), fish_window(["fall", "winter"], ANY, "18:00", "26:00", ["ocean"])], aliases=["长鳍金枪鱼"]),
    fish("Sunfish", [fish_window(["spring", "summer"], ["sunny"], "06:00", "19:00", ["river"])], aliases=["太阳鱼"]),
    fish("Catfish", [fish_window(["spring", "fall"], ["rain"], "06:00", "24:00", ["river", "secret_woods", "witchs_swamp"]), fish_window(["summer"], ["rain"], "06:00", "24:00", ["secret_woods", "witchs_swamp"]), fish_window(["winter"], ["rain_totem"], "06:00", "24:00", ["river", "secret_woods", "witchs_swamp"])], aliases=["鲶鱼"], difficulty=75, behavior="mixed"),
    fish("Shad", [fish_window(["spring", "summer", "fall"], ["rain"], "09:00", "26:00", ["river"])], aliases=["西鲱"]),
    fish("Tiger Trout", [fish_window(["fall", "winter"], ANY, "06:00", "19:00", ["river"])], aliases=["虎纹鳟鱼"]),
    fish("Salmon", [fish_window(["fall"], ANY, "06:00", "19:00", ["river"])], aliases=["鲑鱼"]),
    fish("Rainbow Trout", [fish_window(["summer"], ["sunny"], "06:00", "19:00", ["river", "mountain_lake"])], aliases=["虹鳟鱼"]),
    fish("Lingcod", [fish_window(["winter"], ANY, "06:00", "26:00", ["river", "mountain_lake"])], aliases=["蛇齿单线鱼"]),
    fish("Walleye", [fish_window(["fall"], ["rain"], "12:00", "26:00", ["river", "mountain_lake", "forest_pond"]), fish_window(["winter"], ["rain_totem"], "12:00", "26:00", ["river", "mountain_lake", "forest_pond"])], aliases=["大眼鱼"], difficulty=45, behavior="smooth"),
    fish("Bream", [fish_window(ALL_SEASONS, ANY, "18:00", "26:00", ["river"])], aliases=["鲷鱼"], difficulty=35, behavior="smooth"),
    fish("Smallmouth Bass", [fish_window(["spring", "fall"], ANY, "06:00", "26:00", ["river", "forest_pond"])], aliases=["小嘴鲈鱼"]),
    fish("Dorado", [fish_window(["summer"], ANY, "06:00", "19:00", ["forest_river"])], aliases=["鲯鳅"]),
    fish("Largemouth Bass", [fish_window(ALL_SEASONS, ANY, "06:00", "19:00", ["mountain_lake"])], aliases=["大嘴鲈鱼"]),
    fish("Carp", [fish_window(ALL_SEASONS, ANY, "00:00", "24:00", ["mountain_lake", "secret_woods", "sewers"])], aliases=["鲤鱼"]),
    fish("Bullhead", [fish_window(ALL_SEASONS, ANY, "00:00", "24:00", ["mountain_lake"])], aliases=["大头鱼"], difficulty=46, behavior="smooth"),
    fish("Sturgeon", [fish_window(["summer", "winter"], ANY, "06:00", "19:00", ["mountain_lake"])], aliases=["鲟鱼"], difficulty=78, behavior="mixed"),
    fish("Midnight Carp", [fish_window(["fall", "winter"], ANY, "22:00", "26:00", ["mountain_lake", "forest_pond", "ginger_island_freshwater"])], aliases=["午夜鲤鱼"], difficulty=55, behavior="mixed"),
    fish("Perch", [fish_window(["winter"], ANY, "00:00", "24:00", ["river", "mountain_lake", "forest_pond"])], aliases=["河鲈"]),
    fish("Chub", [fish_window(ALL_SEASONS, ANY, "00:00", "24:00", ["river", "mountain_lake"])], aliases=["鲢鱼"]),
    fish("Ghostfish", [fish_window(ALL_SEASONS, ANY, "00:00", "24:00", ["mines_20", "mines_60"])], aliases=["鬼鱼"], difficulty=50, behavior="mixed"),
    fish("Stonefish", [fish_window(ALL_SEASONS, ANY, "00:00", "24:00", ["mines_20"])], aliases=["石鱼"]),
    fish("Ice Pip", [fish_window(ALL_SEASONS, ANY, "00:00", "24:00", ["mines_60"])], aliases=["冰柱鱼"]),
    fish("Lava Eel", [fish_window(ALL_SEASONS, ANY, "00:00", "24:00", ["mines_100", "volcano_caldera"])], aliases=["岩浆鳗鱼"]),
    fish("Sandfish", [fish_window(ALL_SEASONS, ANY, "06:00", "20:00", ["desert_pond"])], aliases=["沙鱼"], difficulty=65, behavior="mixed"),
    fish("Scorpion Carp", [fish_window(ALL_SEASONS, ANY, "06:00", "20:00", ["desert_pond"])], aliases=["蝎鲤鱼"]),
    fish("Woodskip", [fish_window(ALL_SEASONS, ANY, "00:00", "24:00", ["secret_woods"])], aliases=["木跃鱼"]),
    fish("Void Salmon", [fish_window(ALL_SEASONS, ANY, "00:00", "24:00", ["witchs_swamp"])], aliases=["虚空鲑鱼"]),
    fish("Slimejack", [fish_window(ALL_SEASONS, ANY, "00:00", "24:00", ["mutant_bug_lair"])], aliases=["史莱姆鱼"]),
    fish("Blue Discus", [fish_window(ALL_SEASONS, ANY, "06:00", "26:00", ["ginger_island_freshwater"])], aliases=["蓝铁饼鱼"]),
    fish("Lionfish", [fish_window(ALL_SEASONS, ANY, "06:00", "26:00", ["ginger_island_ocean"])], aliases=["狮子鱼"]),
    fish("Stingray", [fish_window(ALL_SEASONS, ANY, "06:00", "26:00", ["pirate_cove"])], aliases=["黄貂鱼"]),
    fish("Legend", [fish_window(["spring"], ["rain"], "06:00", "20:00", ["mountain_lake"])], aliases=["传说之鱼"], special_conditions=["Requires Fishing level 10 and the mountain-lake log fishing zone."]),
    fish("Crimsonfish", [fish_window(["summer"], ANY, "06:00", "20:00", ["east_pier_ocean"])], aliases=["绯红鱼"], special_conditions=["Requires Fishing level 5 and access to the east pier."]),
    fish("Angler", [fish_window(["fall"], ANY, "06:00", "26:00", ["north_town_river"])], aliases=["鮟鱇鱼"], special_conditions=["Caught north of the wooden plank bridge near JojaMart."]),
    fish("Glacierfish", [fish_window(["winter"], ANY, "06:00", "20:00", ["cindersap_forest_south"])], aliases=["冰川鱼"], special_conditions=["Requires Fishing level 6 and the southern island fishing zone."]),
    fish("Mutant Carp", [fish_window(ALL_SEASONS, ANY, "00:00", "24:00", ["sewers"])], aliases=["变种鲤鱼"]),
    fish("Blobfish", [fish_window(["winter"], ANY, "17:00", "26:00", ["night_market_submarine"])], aliases=["水滴鱼"], special_conditions=["Available during the Night Market submarine ride on Winter 15-17."]),
    fish("Midnight Squid", [fish_window(["winter"], ANY, "17:00", "26:00", ["night_market_submarine"])], aliases=["午夜鱿鱼"], special_conditions=["Available during the Night Market submarine ride on Winter 15-17."]),
    fish("Spook Fish", [fish_window(["winter"], ANY, "17:00", "26:00", ["night_market_submarine"])], aliases=["幽灵鱼"], special_conditions=["Available during the Night Market submarine ride on Winter 15-17."]),
    fish("Goby", [fish_window(ALL_SEASONS, ANY, "06:00", "26:00", ["cindersap_forest_waterfall"])], aliases=["虾虎鱼"], special_conditions=["Caught in the waterfalls in southern Cindersap Forest."]),
]


def villager(
    name: str,
    season: str,
    day: int,
    loved: list[str],
    *,
    aliases: Iterable[str] = (),
    home: str | None = None,
    marriageable: bool = False,
) -> dict[str, Any]:
    return base_record(
        "villager",
        name,
        aliases=aliases,
        facts={
            "birthday": {"season": season, "day": day},
            "marriageable": marriageable,
            "home": home,
            "loved_gifts": loved,
            "gift_rules": {"universal_loves_apply": True, "personal_exceptions_tracked": False},
        },
        page_title=name,
        section_title="Gifts",
    )


VILLAGERS = [
    villager("Abigail", "fall", 13, ["Amethyst", "Banana Pudding", "Blackberry Cobbler", "Chocolate Cake", "Pufferfish", "Pumpkin", "Spicy Eel"], aliases=["阿比盖尔"], home="Pierre's General Store", marriageable=True),
    villager("Alex", "summer", 13, ["Complete Breakfast", "Salmon Dinner"], aliases=["亚历克斯"], home="1 River Road", marriageable=True),
    villager("Elliott", "fall", 5, ["Crab Cakes", "Duck Feather", "Lobster", "Pomegranate", "Squid Ink", "Tom Kha Soup"], aliases=["艾利欧特"], home="Elliott's Cabin", marriageable=True),
    villager("Emily", "spring", 27, ["Amethyst", "Aquamarine", "Cloth", "Emerald", "Jade", "Ruby", "Survival Burger", "Topaz", "Wool"], aliases=["艾米丽"], home="2 Willow Lane", marriageable=True),
    villager("Haley", "spring", 14, ["Coconut", "Fruit Salad", "Pink Cake", "Sunflower"], aliases=["海莉"], home="2 Willow Lane", marriageable=True),
    villager("Harvey", "winter", 14, ["Coffee", "Pickles", "Super Meal", "Truffle Oil", "Wine"], aliases=["哈维"], home="Harvey's Clinic", marriageable=True),
    villager("Leah", "winter", 23, ["Goat Cheese", "Poppyseed Muffin", "Salad", "Stir Fry", "Truffle", "Vegetable Medley", "Wine"], aliases=["莉亚"], home="Leah's Cottage", marriageable=True),
    villager("Maru", "summer", 10, ["Battery Pack", "Cauliflower", "Cheese Cauliflower", "Diamond", "Gold Bar", "Iridium Bar", "Miner's Treat", "Pepper Poppers", "Rhubarb Pie", "Strawberry"], aliases=["玛鲁"], home="Carpenter's Shop", marriageable=True),
    villager("Penny", "fall", 2, ["Diamond", "Emerald", "Melon", "Poppy", "Poppyseed Muffin", "Red Plate", "Roots Platter", "Sandfish", "Tom Kha Soup"], aliases=["潘妮"], home="Trailer", marriageable=True),
    villager("Sam", "summer", 17, ["Cactus Fruit", "Maple Bar", "Pizza", "Tigerseye"], aliases=["山姆"], home="1 Willow Lane", marriageable=True),
    villager("Sebastian", "winter", 10, ["Frozen Tear", "Obsidian", "Pumpkin Soup", "Sashimi", "Void Egg"], aliases=["塞巴斯蒂安"], home="Carpenter's Shop", marriageable=True),
    villager("Shane", "spring", 20, ["Beer", "Hot Pepper", "Pepper Poppers", "Pizza"], aliases=["谢恩"], home="Marnie's Ranch", marriageable=True),
    villager("Caroline", "winter", 7, ["Fish Taco", "Green Tea", "Summer Spangle", "Tropical Curry"], aliases=["卡洛琳"], home="Pierre's General Store"),
    villager("Clint", "winter", 26, ["Amethyst", "Aquamarine", "Artichoke Dip", "Emerald", "Fiddlehead Risotto", "Gold Bar", "Iridium Bar", "Jade", "Omni Geode", "Ruby", "Topaz"], aliases=["克林特"], home="Blacksmith"),
    villager("Demetrius", "summer", 19, ["Bean Hotpot", "Ice Cream", "Rice Pudding", "Strawberry"], aliases=["德米特里厄斯"], home="Carpenter's Shop"),
    villager("Evelyn", "winter", 20, ["Beet", "Chocolate Cake", "Diamond", "Fairy Rose", "Stuffing", "Tulip"], aliases=["艾芙琳"], home="1 River Road"),
    villager("George", "fall", 24, ["Fried Mushroom", "Leek"], aliases=["乔治"], home="1 River Road"),
    villager("Gus", "summer", 8, ["Diamond", "Escargot", "Fish Taco", "Orange", "Tropical Curry"], aliases=["格斯"], home="The Stardrop Saloon"),
    villager("Jas", "summer", 4, ["Fairy Rose", "Pink Cake", "Plum Pudding"], aliases=["贾斯"], home="Marnie's Ranch"),
    villager("Jodi", "fall", 11, ["Chocolate Cake", "Crispy Bass", "Diamond", "Eggplant Parmesan", "Fried Eel", "Pancakes", "Rhubarb Pie", "Vegetable Medley"], aliases=["乔迪"], home="1 Willow Lane"),
    villager("Kent", "spring", 4, ["Fiddlehead Risotto", "Roasted Hazelnuts"], aliases=["肯特"], home="1 Willow Lane"),
    villager("Lewis", "spring", 7, ["Autumn's Bounty", "Glazed Yams", "Green Tea", "Hot Pepper", "Vegetable Medley"], aliases=["刘易斯"], home="Mayor's Manor"),
    villager("Linus", "winter", 3, ["Blueberry Tart", "Cactus Fruit", "Coconut", "Dish O' The Sea", "Yam"], aliases=["莱纳斯"], home="Tent"),
    villager("Marnie", "fall", 18, ["Diamond", "Farmer's Lunch", "Pink Cake", "Pumpkin Pie"], aliases=["玛妮"], home="Marnie's Ranch"),
    villager("Pam", "spring", 18, ["Beer", "Cactus Fruit", "Glazed Yams", "Mead", "Pale Ale", "Parsnip", "Parsnip Soup"], aliases=["潘姆"], home="Trailer"),
    villager("Pierre", "spring", 26, ["Fried Calamari"], aliases=["皮埃尔"], home="Pierre's General Store"),
    villager("Robin", "fall", 21, ["Goat Cheese", "Peach", "Spaghetti"], aliases=["罗宾"], home="Carpenter's Shop"),
    villager("Vincent", "spring", 10, ["Cranberry Candy", "Ginger Ale", "Grape", "Pink Cake", "Snail"], aliases=["文森特"], home="1 Willow Lane"),
    villager("Willy", "summer", 24, ["Catfish", "Diamond", "Iridium Bar", "Mead", "Octopus", "Pumpkin", "Sea Cucumber", "Sturgeon"], aliases=["威利"], home="Fish Shop"),
    villager("Wizard", "winter", 17, ["Purple Mushroom", "Solar Essence", "Super Cucumber", "Void Essence"], aliases=["法师"], home="Wizard's Tower"),
    villager("Leo", "summer", 26, ["Duck Feather", "Mango", "Ostrich Egg", "Poi"], aliases=["雷欧"], home="Treehouse"),
    villager("Sandy", "fall", 15, ["Crocus", "Daffodil", "Mango Sticky Rice", "Sweet Pea"], aliases=["桑迪"], home="Oasis"),
    villager("Krobus", "winter", 1, ["Diamond", "Iridium Bar", "Pumpkin", "Void Egg", "Void Mayonnaise", "Wild Horseradish"], aliases=["科罗布斯"], home="The Sewers"),
    villager("Dwarf", "summer", 22, ["Amethyst", "Aquamarine", "Emerald", "Jade", "Lemon Stone", "Omni Geode", "Ruby", "Topaz"], aliases=["矮人"], home="The Mines"),
]


def ingredient(name: str, quantity: int = 1, *, category_substitution: str | None = None) -> dict[str, Any]:
    return {
        "item_name": name,
        "quantity": quantity,
        "category_substitution": category_substitution,
    }


def recipe(
    name: str,
    recipe_type: str,
    ingredients: list[tuple[str, int]],
    *,
    aliases: Iterable[str] = (),
    unlock_source: str,
    skill: str | None = None,
    level: int | None = None,
    result_quantity: int = 1,
    effect: str | None = None,
) -> dict[str, Any]:
    unlock: dict[str, Any] = {"source": unlock_source}
    if skill is not None:
        unlock["skill"] = skill
    if level is not None:
        unlock["level"] = level
    facts: dict[str, Any] = {
        "recipe_type": recipe_type,
        "result_name": name,
        "result_quantity": result_quantity,
        "ingredients": [
            ingredient(item, quantity, category_substitution=item if item.startswith("Any ") else None)
            for item, quantity in ingredients
        ],
        "unlock": unlock,
        "unlock_source": unlock_source,
        "skill_requirement": {"skill": skill, "level": level} if skill is not None else None,
        "friendship_requirement": None,
        "purchase_source": None,
        "purchase_price": None,
        "version_notes": [],
    }
    if effect:
        facts["effect"] = effect
    return base_record(
        "recipe",
        name,
        aliases=aliases,
        facts=facts,
        page_title=name,
        section_title="Cooking" if recipe_type == "cooking" else "Crafting",
    )


COOKING_SPECS: list[tuple[str, list[tuple[str, int]], str]] = [
    ("Fried Egg", [("Any Egg", 1)], "Upgraded farmhouse kitchen"),
    ("Omelet", [("Any Egg", 1), ("Milk", 1)], "The Queen of Sauce, 28 Spring Year 1; or The Stardrop Saloon"),
    ("Salad", [("Leek", 1), ("Dandelion", 1), ("Vinegar", 1)], "Emily at 3 hearts"),
    ("Cheese Cauliflower", [("Cauliflower", 1), ("Cheese", 1)], "Pam at 3 hearts"),
    ("Baked Fish", [("Sunfish", 1), ("Bream", 1), ("Wheat Flour", 1)], "The Queen of Sauce, 7 Summer Year 1"),
    ("Parsnip Soup", [("Parsnip", 1), ("Milk", 1), ("Vinegar", 1)], "Caroline at 3 hearts"),
    ("Vegetable Medley", [("Tomato", 1), ("Beet", 1)], "Caroline at 7 hearts"),
    ("Complete Breakfast", [("Fried Egg", 1), ("Milk", 1), ("Hashbrowns", 1), ("Pancakes", 1)], "The Queen of Sauce, 21 Spring Year 2"),
    ("Fried Calamari", [("Squid", 1), ("Wheat Flour", 1), ("Oil", 1)], "Jodi at 3 hearts"),
    ("Strange Bun", [("Wheat Flour", 1), ("Periwinkle", 1), ("Void Mayonnaise", 1)], "Shane at 7 hearts"),
    ("Lucky Lunch", [("Sea Cucumber", 1), ("Tortilla", 1), ("Blue Jazz", 1)], "The Queen of Sauce, 28 Spring Year 2"),
    ("Fried Mushroom", [("Common Mushroom", 1), ("Morel", 1), ("Oil", 1)], "Demetrius at 3 hearts"),
    ("Pizza", [("Wheat Flour", 1), ("Tomato", 1), ("Cheese", 1)], "The Queen of Sauce; or The Stardrop Saloon"),
    ("Bean Hotpot", [("Green Bean", 2)], "Clint at 7 hearts"),
    ("Glazed Yams", [("Yam", 1), ("Sugar", 1)], "The Queen of Sauce, 21 Fall Year 1"),
    ("Carp Surprise", [("Carp", 4)], "The Queen of Sauce, 7 Summer Year 2"),
    ("Hashbrowns", [("Potato", 1), ("Oil", 1)], "The Queen of Sauce; or The Stardrop Saloon"),
    ("Pancakes", [("Wheat Flour", 1), ("Any Egg", 1)], "The Queen of Sauce; or The Stardrop Saloon"),
    ("Salmon Dinner", [("Salmon", 1), ("Amaranth", 1), ("Kale", 1)], "Gus at 3 hearts"),
    ("Fish Taco", [("Tuna", 1), ("Tortilla", 1), ("Red Cabbage", 1), ("Mayonnaise", 1)], "Linus at 7 hearts"),
    ("Crispy Bass", [("Largemouth Bass", 1), ("Wheat Flour", 1), ("Oil", 1)], "Kent at 3 hearts"),
    ("Pepper Poppers", [("Hot Pepper", 1), ("Cheese", 1)], "Shane at 3 hearts"),
    ("Bread", [("Wheat Flour", 1)], "The Queen of Sauce; or The Stardrop Saloon"),
    ("Tom Kha Soup", [("Coconut", 1), ("Shrimp", 1), ("Common Mushroom", 1)], "Sandy at 7 hearts"),
    ("Trout Soup", [("Rainbow Trout", 1), ("Green Algae", 1)], "The Queen of Sauce"),
    ("Chocolate Cake", [("Wheat Flour", 1), ("Sugar", 1), ("Any Egg", 1)], "The Queen of Sauce, 14 Winter Year 1"),
    ("Pink Cake", [("Melon", 1), ("Wheat Flour", 1), ("Sugar", 1), ("Any Egg", 1)], "The Queen of Sauce, 21 Summer Year 2"),
    ("Rhubarb Pie", [("Rhubarb", 1), ("Wheat Flour", 1), ("Sugar", 1)], "Marnie at 7 hearts"),
    ("Cookie", [("Wheat Flour", 1), ("Sugar", 1), ("Any Egg", 1)], "Evelyn at 4 hearts"),
    ("Spaghetti", [("Wheat Flour", 1), ("Tomato", 1)], "Lewis at 3 hearts"),
    ("Fried Eel", [("Eel", 1), ("Oil", 1)], "George at 3 hearts"),
    ("Spicy Eel", [("Eel", 1), ("Hot Pepper", 1)], "George at 7 hearts"),
    ("Sashimi", [("Any Fish", 1)], "Linus at 3 hearts"),
    ("Maki Roll", [("Any Fish", 1), ("Seaweed", 1), ("Rice", 1)], "The Queen of Sauce; or The Stardrop Saloon"),
    ("Tortilla", [("Corn", 1)], "The Queen of Sauce; or The Stardrop Saloon"),
    ("Red Plate", [("Red Cabbage", 1), ("Radish", 1)], "Emily at 7 hearts"),
    ("Eggplant Parmesan", [("Eggplant", 1), ("Tomato", 1)], "Lewis at 7 hearts"),
    ("Rice Pudding", [("Milk", 1), ("Sugar", 1), ("Rice", 1)], "Evelyn at 7 hearts"),
    ("Ice Cream", [("Milk", 1), ("Sugar", 1)], "Jodi at 7 hearts"),
    ("Blueberry Tart", [("Blueberry", 1), ("Wheat Flour", 1), ("Sugar", 1), ("Any Egg", 1)], "Pierre at 3 hearts"),
    ("Autumn's Bounty", [("Yam", 1), ("Pumpkin", 1)], "Demetrius at 7 hearts"),
    ("Pumpkin Soup", [("Pumpkin", 1), ("Milk", 1)], "Robin at 7 hearts"),
    ("Super Meal", [("Bok Choy", 1), ("Cranberries", 1), ("Artichoke", 1)], "Kent at 7 hearts"),
    ("Cranberry Sauce", [("Cranberries", 1), ("Sugar", 1)], "The Queen of Sauce, 28 Fall Year 1"),
    ("Stuffing", [("Bread", 1), ("Cranberries", 1), ("Hazelnut", 1)], "Pam at 7 hearts"),
    ("Farmer's Lunch", [("Omelet", 1), ("Parsnip", 1)], "Farming level 3"),
    ("Survival Burger", [("Bread", 1), ("Cave Carrot", 1), ("Eggplant", 1)], "Foraging level 2"),
    ("Dish O' The Sea", [("Sardine", 2), ("Hashbrowns", 1)], "Fishing level 3"),
    ("Miner's Treat", [("Cave Carrot", 2), ("Sugar", 1), ("Milk", 1)], "Mining level 3"),
    ("Roots Platter", [("Cave Carrot", 1), ("Winter Root", 1)], "Combat level 3"),
    ("Triple Shot Espresso", [("Coffee", 3)], "The Stardrop Saloon"),
    ("Seafoam Pudding", [("Flounder", 1), ("Midnight Carp", 1), ("Squid Ink", 1)], "Fishing level 9"),
    ("Algae Soup", [("Green Algae", 4)], "Clint at 3 hearts"),
    ("Pale Broth", [("White Algae", 2)], "Marnie at 3 hearts"),
    ("Plum Pudding", [("Wild Plum", 2), ("Wheat Flour", 1), ("Sugar", 1)], "The Queen of Sauce, 7 Winter Year 1"),
    ("Artichoke Dip", [("Artichoke", 1), ("Milk", 1)], "The Queen of Sauce, 28 Fall Year 1"),
    ("Stir Fry", [("Cave Carrot", 1), ("Common Mushroom", 1), ("Kale", 1), ("Oil", 1)], "The Queen of Sauce, 7 Spring Year 1"),
    ("Roasted Hazelnuts", [("Hazelnut", 3)], "The Queen of Sauce, 28 Summer Year 2"),
    ("Pumpkin Pie", [("Pumpkin", 1), ("Wheat Flour", 1), ("Milk", 1), ("Sugar", 1)], "The Queen of Sauce, 21 Winter Year 1"),
    ("Radish Salad", [("Oil", 1), ("Vinegar", 1), ("Radish", 1)], "The Queen of Sauce, 21 Spring Year 1"),
    ("Fruit Salad", [("Blueberry", 1), ("Melon", 1), ("Apricot", 1)], "The Queen of Sauce, 7 Fall Year 2"),
    ("Blackberry Cobbler", [("Blackberry", 2), ("Sugar", 1), ("Wheat Flour", 1)], "The Queen of Sauce, 14 Fall Year 2"),
    ("Cranberry Candy", [("Cranberries", 1), ("Apple", 1), ("Sugar", 1)], "The Queen of Sauce, 28 Winter Year 1"),
    ("Bruschetta", [("Bread", 1), ("Oil", 1), ("Tomato", 1)], "The Queen of Sauce, 21 Winter Year 2"),
    ("Coleslaw", [("Red Cabbage", 1), ("Vinegar", 1), ("Mayonnaise", 1)], "The Queen of Sauce, 14 Spring Year 1"),
    ("Fiddlehead Risotto", [("Oil", 1), ("Fiddlehead Fern", 1), ("Garlic", 1)], "The Queen of Sauce, 28 Fall Year 2"),
    ("Poppyseed Muffin", [("Poppy", 1), ("Wheat Flour", 1), ("Sugar", 1)], "The Queen of Sauce, 7 Winter Year 2"),
    ("Chowder", [("Clam", 1), ("Milk", 1)], "Willy at 3 hearts"),
    ("Fish Stew", [("Crayfish", 1), ("Mussel", 1), ("Periwinkle", 1), ("Tomato", 1)], "Willy at 7 hearts"),
    ("Escargot", [("Snail", 1), ("Garlic", 1)], "Willy at 5 hearts"),
    ("Lobster Bisque", [("Lobster", 1), ("Milk", 1)], "Willy at 9 hearts; or The Queen of Sauce"),
    ("Maple Bar", [("Maple Syrup", 1), ("Sugar", 1), ("Wheat Flour", 1)], "The Queen of Sauce, 14 Summer Year 2"),
    ("Crab Cakes", [("Crab", 1), ("Wheat Flour", 1), ("Any Egg", 1), ("Oil", 1)], "The Queen of Sauce, 21 Fall Year 2"),
    ("Shrimp Cocktail", [("Shrimp", 1), ("Tomato", 1), ("Wild Horseradish", 1)], "The Queen of Sauce, 28 Winter Year 2"),
    ("Ginger Ale", [("Ginger", 3), ("Sugar", 1)], "Dwarf Shop in Volcano Dungeon"),
    ("Banana Pudding", [("Banana", 1), ("Milk", 1), ("Sugar", 1)], "Island Trader"),
    ("Mango Sticky Rice", [("Mango", 1), ("Coconut", 1), ("Rice", 1)], "Leo at 7 hearts"),
    ("Poi", [("Taro Root", 4)], "Leo at 3 hearts"),
    ("Tropical Curry", [("Coconut", 1), ("Pineapple", 1), ("Hot Pepper", 1)], "Gus at the Ginger Island Resort"),
    ("Squid Ink Ravioli", [("Squid Ink", 1), ("Wheat Flour", 1), ("Tomato", 1)], "Combat level 9"),
]

COOKING_ALIASES = {
    "Fried Egg": ["荷包蛋"], "Omelet": ["煎蛋卷"], "Salad": ["沙拉"],
    "Complete Breakfast": ["完整早餐"], "Pizza": ["披萨"], "Bread": ["面包"],
    "Chocolate Cake": ["巧克力蛋糕"], "Pink Cake": ["粉红蛋糕"], "Spaghetti": ["意大利面"],
    "Sashimi": ["生鱼片"], "Maki Roll": ["生鱼寿司"], "Pumpkin Soup": ["南瓜汤"],
    "Triple Shot Espresso": ["三倍浓缩咖啡"], "Ginger Ale": ["姜汁汽水"],
}

COOKING = [
    recipe(name, "cooking", ingredients, aliases=COOKING_ALIASES.get(name, []), unlock_source=source)
    for name, ingredients, source in COOKING_SPECS
]

CRAFTING_SPECS: list[tuple[str, list[tuple[str, int]], str, str | None, int | None, str | None]] = [
    ("Chest", [("Wood", 50)], "Starter recipe", None, None, "Stores items."),
    ("Wood Fence", [("Wood", 2)], "Starter recipe", None, None, "Blocks movement and contains animals."),
    ("Gate", [("Wood", 10)], "Starter recipe", None, None, "Allows passage through fences."),
    ("Torch", [("Wood", 1), ("Sap", 2)], "Starter recipe", None, None, "Produces light."),
    ("Furnace", [("Copper Ore", 20), ("Stone", 25)], "Clint after obtaining Copper Ore", None, None, "Smelts ore into bars."),
    ("Scarecrow", [("Wood", 50), ("Coal", 1), ("Fiber", 20)], "Farming level 1", "farming", 1, "Protects crops from crows."),
    ("Basic Fertilizer", [("Sap", 2)], "Farming level 1", "farming", 1, None),
    ("Speed-Gro", [("Pine Tar", 1), ("Clam", 1)], "Farming level 3", "farming", 3, None),
    ("Quality Fertilizer", [("Sap", 4), ("Any Fish", 1)], "Farming level 9", "farming", 9, None),
    ("Sprinkler", [("Copper Bar", 1), ("Iron Bar", 1)], "Farming level 2", "farming", 2, "Waters the 4 adjacent tiles every morning."),
    ("Quality Sprinkler", [("Iron Bar", 1), ("Gold Bar", 1), ("Refined Quartz", 1)], "Farming level 6", "farming", 6, "Waters the 8 adjacent tiles every morning."),
    ("Iridium Sprinkler", [("Gold Bar", 1), ("Iridium Bar", 1), ("Battery Pack", 1)], "Farming level 9", "farming", 9, "Waters 24 adjacent tiles every morning."),
    ("Bee House", [("Wood", 40), ("Coal", 8), ("Iron Bar", 1), ("Maple Syrup", 1)], "Farming level 3", "farming", 3, "Produces Honey outdoors outside Winter."),
    ("Keg", [("Wood", 30), ("Copper Bar", 1), ("Iron Bar", 1), ("Oak Resin", 1)], "Farming level 8", "farming", 8, "Processes fruit, vegetables, wheat, hops, coffee beans, and tea leaves."),
    ("Preserves Jar", [("Wood", 50), ("Stone", 40), ("Coal", 8)], "Farming level 4", "farming", 4, "Produces Jelly or Pickles."),
    ("Mayonnaise Machine", [("Wood", 15), ("Stone", 15), ("Earth Crystal", 1), ("Copper Bar", 1)], "Farming level 2", "farming", 2, "Processes eggs into Mayonnaise."),
    ("Cheese Press", [("Wood", 45), ("Stone", 45), ("Hardwood", 10), ("Copper Bar", 1)], "Farming level 6", "farming", 6, "Processes milk into Cheese."),
    ("Loom", [("Wood", 60), ("Fiber", 30), ("Pine Tar", 1)], "Farming level 7", "farming", 7, "Processes Wool into Cloth."),
    ("Oil Maker", [("Slime", 50), ("Hardwood", 20), ("Gold Bar", 1)], "Farming level 8", "farming", 8, "Processes Truffles and oil crops."),
    ("Seed Maker", [("Wood", 25), ("Coal", 10), ("Gold Bar", 1)], "Farming level 9", "farming", 9, "Produces seeds from most crops."),
    ("Cask", [("Wood", 20), ("Hardwood", 1)], "Unlocked with the farmhouse cellar", None, None, "Ages selected artisan goods."),
    ("Tapper", [("Wood", 40), ("Copper Bar", 2)], "Foraging level 3", "foraging", 3, "Collects products from trees."),
    ("Charcoal Kiln", [("Wood", 20), ("Copper Bar", 2)], "Foraging level 4", "foraging", 4, "Turns Wood into Coal."),
    ("Lightning Rod", [("Iron Bar", 1), ("Refined Quartz", 1), ("Bat Wing", 5)], "Foraging level 6", "foraging", 6, "Produces Battery Packs from storms."),
    ("Crystalarium", [("Stone", 99), ("Gold Bar", 5), ("Iridium Bar", 2), ("Battery Pack", 1)], "Mining level 9", "mining", 9, "Replicates inserted gems and minerals."),
    ("Cherry Bomb", [("Copper Ore", 4), ("Coal", 1)], "Mining level 1", "mining", 1, None),
    ("Bomb", [("Iron Ore", 4), ("Coal", 1)], "Mining level 6", "mining", 6, None),
    ("Mega Bomb", [("Gold Ore", 4), ("Solar Essence", 1), ("Void Essence", 1)], "Mining level 8", "mining", 8, None),
    ("Staircase", [("Stone", 99)], "Mining level 2", "mining", 2, "Creates a ladder down in mine areas."),
    ("Recycling Machine", [("Wood", 25), ("Stone", 25), ("Iron Bar", 1)], "Fishing level 4", "fishing", 4, "Turns fishing trash into resources."),
    ("Crab Pot", [("Wood", 40), ("Iron Bar", 3)], "Fishing level 3", "fishing", 3, "Catches crab-pot fish when baited."),
    ("Worm Bin", [("Hardwood", 25), ("Gold Bar", 1), ("Iron Bar", 1), ("Fiber", 50)], "Fishing level 8", "fishing", 8, "Produces Bait."),
    ("Slime Egg-Press", [("Coal", 25), ("Fire Quartz", 1), ("Battery Pack", 1)], "Combat level 6", "combat", 6, "Compresses Slime into Slime Eggs."),
    ("Slime Incubator", [("Iridium Bar", 2), ("Slime", 100)], "Combat level 8", "combat", 8, "Hatches Slime Eggs."),
    ("Bone Mill", [("Bone Fragment", 10), ("Clay", 3), ("Stone", 20)], "Gunther's Fragments of the Past special order", None, None, "Turns bone items into fertilizer."),
    ("Dehydrator", [("Wood", 30), ("Clay", 2), ("Fire Quartz", 1)], "Purchase the recipe from Pierre", None, None, "Dries fruit and mushrooms."),
    ("Fish Smoker", [("Hardwood", 10), ("Sea Jelly", 1), ("River Jelly", 1), ("Cave Jelly", 1)], "Purchase the recipe from Willy", None, None, "Smokes fish while retaining quality."),
]

CRAFTING_ALIASES = {
    "Chest": ["箱子"], "Furnace": ["熔炉"], "Scarecrow": ["稻草人"],
    "Sprinkler": ["洒水器"], "Quality Sprinkler": ["优质洒水器"],
    "Iridium Sprinkler": ["铱制洒水器"], "Keg": ["小桶"], "Preserves Jar": ["罐头瓶"],
    "Mayonnaise Machine": ["蛋黄酱机"], "Cheese Press": ["压酪机"], "Loom": ["织布机"],
    "Oil Maker": ["产油机"], "Seed Maker": ["种子生产器"], "Crystalarium": ["宝石复制机"],
    "Crab Pot": ["蟹笼"], "Slime Incubator": ["史莱姆孵化器"], "Fish Smoker": ["熏鱼机"],
}

CRAFTING = [
    recipe(
        name,
        "crafting",
        ingredients,
        aliases=CRAFTING_ALIASES.get(name, []),
        unlock_source=source,
        skill=skill,
        level=level,
        effect=effect,
    )
    for name, ingredients, source, skill, level, effect in CRAFTING_SPECS
]

RECIPES = COOKING + CRAFTING


def requirement(
    item_name: str,
    quantity: int = 1,
    *,
    minimum_quality: str | None = None,
    optional_group: str | None = None,
) -> dict[str, Any]:
    return {
        "item_name": item_name,
        "quantity": quantity,
        "minimum_quality": minimum_quality,
        "optional_group": optional_group,
    }


def bundle(
    name: str,
    room: str,
    requirements: list[dict[str, Any]],
    *,
    aliases: Iterable[str] = (),
    selection_rule: str = "all",
    reward: str | None = None,
    room_reward: str | None = None,
) -> dict[str, Any]:
    return base_record(
        "bundle",
        name,
        aliases=aliases,
        facts={
            "bundle_mode": "standard",
            "room": room,
            "bundle_name": name,
            "selection_rule": selection_rule,
            "requirements": requirements,
            "reward": reward,
            "room_reward": room_reward,
            "route": "community_center",
        },
        conditions={"route": "community_center", "bundle_mode": "standard"},
        page_title="Bundles",
        section_title=f"Standard Bundles > {room}",
    )


BUNDLES = [
    bundle("Spring Foraging Bundle", "Crafts Room", [requirement("Wild Horseradish"), requirement("Daffodil"), requirement("Leek"), requirement("Dandelion")], aliases=["春季觅食收集包"], reward="Spring Seeds (30)", room_reward="Bridge Repair"),
    bundle("Summer Foraging Bundle", "Crafts Room", [requirement("Grape"), requirement("Spice Berry"), requirement("Sweet Pea")], aliases=["夏季觅食收集包"], reward="Summer Seeds (30)", room_reward="Bridge Repair"),
    bundle("Fall Foraging Bundle", "Crafts Room", [requirement("Common Mushroom"), requirement("Wild Plum"), requirement("Hazelnut"), requirement("Blackberry")], aliases=["秋季觅食收集包"], reward="Fall Seeds (30)", room_reward="Bridge Repair"),
    bundle("Winter Foraging Bundle", "Crafts Room", [requirement("Winter Root"), requirement("Crystal Fruit"), requirement("Snow Yam"), requirement("Crocus")], aliases=["冬季觅食收集包"], reward="Winter Seeds (30)", room_reward="Bridge Repair"),
    bundle("Construction Bundle", "Crafts Room", [requirement("Wood", 99), requirement("Wood", 99), requirement("Stone", 99), requirement("Hardwood", 10)], aliases=["建筑收集包"], reward="Charcoal Kiln", room_reward="Bridge Repair"),
    bundle("Exotic Foraging Bundle", "Crafts Room", [requirement(item, optional_group="choose_any_5") for item in ["Coconut", "Cactus Fruit", "Cave Carrot", "Red Mushroom", "Purple Mushroom", "Maple Syrup", "Oak Resin", "Pine Tar", "Morel"]], aliases=["奇异觅食收集包"], selection_rule="choose_5_of_9", reward="Autumn's Bounty (5)", room_reward="Bridge Repair"),
    bundle("Spring Crops Bundle", "Pantry", [requirement("Parsnip"), requirement("Green Bean"), requirement("Cauliflower"), requirement("Potato")], aliases=["春季作物收集包"], reward="Speed-Gro (20)", room_reward="Greenhouse"),
    bundle("Summer Crops Bundle", "Pantry", [requirement("Tomato"), requirement("Hot Pepper"), requirement("Blueberry"), requirement("Melon")], aliases=["夏季作物收集包"], reward="Quality Sprinkler", room_reward="Greenhouse"),
    bundle("Fall Crops Bundle", "Pantry", [requirement("Corn"), requirement("Eggplant"), requirement("Pumpkin"), requirement("Yam")], aliases=["秋季作物收集包"], reward="Bee House", room_reward="Greenhouse"),
    bundle("Quality Crops Bundle", "Pantry", [requirement("Parsnip", 5, minimum_quality="gold", optional_group="choose_any_3"), requirement("Melon", 5, minimum_quality="gold", optional_group="choose_any_3"), requirement("Pumpkin", 5, minimum_quality="gold", optional_group="choose_any_3"), requirement("Corn", 5, minimum_quality="gold", optional_group="choose_any_3")], aliases=["高品质作物收集包"], selection_rule="choose_3_of_4", reward="Preserves Jar", room_reward="Greenhouse"),
    bundle("Animal Bundle", "Pantry", [requirement(item, optional_group="choose_any_5") for item in ["Large Milk", "Large Brown Egg", "Large Egg", "Large Goat Milk", "Wool", "Duck Egg"]], aliases=["动物制品收集包"], selection_rule="choose_5_of_6", reward="Cheese Press", room_reward="Greenhouse"),
    bundle("Artisan Bundle", "Pantry", [requirement(item, optional_group="choose_any_6") for item in ["Truffle Oil", "Cloth", "Goat Cheese", "Cheese", "Honey", "Jelly", "Apple", "Apricot", "Orange", "Peach", "Pomegranate", "Cherry"]], aliases=["工匠物品收集包"], selection_rule="choose_6_of_12", reward="Keg", room_reward="Greenhouse"),
    bundle("River Fish Bundle", "Fish Tank", [requirement("Sunfish"), requirement("Catfish"), requirement("Shad"), requirement("Tiger Trout")], aliases=["河鱼收集包"], reward="Bait (30)", room_reward="Glittering Boulder Removed"),
    bundle("Lake Fish Bundle", "Fish Tank", [requirement("Largemouth Bass"), requirement("Carp"), requirement("Bullhead"), requirement("Sturgeon")], aliases=["湖鱼收集包"], reward="Dressed Spinner", room_reward="Glittering Boulder Removed"),
    bundle("Ocean Fish Bundle", "Fish Tank", [requirement("Sardine"), requirement("Tuna"), requirement("Red Snapper"), requirement("Tilapia")], aliases=["海鱼收集包"], reward="Warp Totem: Beach (5)", room_reward="Glittering Boulder Removed"),
    bundle("Night Fishing Bundle", "Fish Tank", [requirement("Walleye"), requirement("Bream"), requirement("Eel")], aliases=["夜间垂钓收集包"], reward="Small Glow Ring", room_reward="Glittering Boulder Removed"),
    bundle("Crab Pot Bundle", "Fish Tank", [requirement(item, optional_group="choose_any_5") for item in ["Lobster", "Crayfish", "Crab", "Cockle", "Mussel", "Shrimp", "Snail", "Periwinkle", "Oyster", "Clam"]], aliases=["蟹笼收集包"], selection_rule="choose_5_of_10", reward="Crab Pot (3)", room_reward="Glittering Boulder Removed"),
    bundle("Specialty Fish Bundle", "Fish Tank", [requirement("Pufferfish"), requirement("Ghostfish"), requirement("Sandfish"), requirement("Woodskip")], aliases=["特色鱼类收集包"], reward="Dish O' The Sea (5)", room_reward="Glittering Boulder Removed"),
    bundle("Blacksmith's Bundle", "Boiler Room", [requirement("Copper Bar"), requirement("Iron Bar"), requirement("Gold Bar")], aliases=["铁匠收集包"], reward="Furnace", room_reward="Minecarts Repaired"),
    bundle("Geologist's Bundle", "Boiler Room", [requirement("Quartz"), requirement("Earth Crystal"), requirement("Frozen Tear"), requirement("Fire Quartz")], aliases=["地质学家收集包"], reward="Omni Geode (5)", room_reward="Minecarts Repaired"),
    bundle("Adventurer's Bundle", "Boiler Room", [requirement("Slime", 99, optional_group="choose_any_2"), requirement("Bat Wing", 10, optional_group="choose_any_2"), requirement("Solar Essence", 1, optional_group="choose_any_2"), requirement("Void Essence", 1, optional_group="choose_any_2")], aliases=["冒险者收集包"], selection_rule="choose_2_of_4", reward="Small Magnet Ring", room_reward="Minecarts Repaired"),
    bundle("Chef's Bundle", "Bulletin Board", [requirement("Maple Syrup"), requirement("Fiddlehead Fern"), requirement("Truffle"), requirement("Poppy"), requirement("Maki Roll"), requirement("Fried Egg")], aliases=["厨师收集包"], reward="Pink Cake (3)", room_reward="Friendship"),
    bundle("Dye Bundle", "Bulletin Board", [requirement("Red Mushroom"), requirement("Sea Urchin"), requirement("Sunflower"), requirement("Duck Feather"), requirement("Aquamarine"), requirement("Red Cabbage")], aliases=["染料收集包"], reward="Seed Maker", room_reward="Friendship"),
    bundle("Field Research Bundle", "Bulletin Board", [requirement("Purple Mushroom"), requirement("Nautilus Shell"), requirement("Chub"), requirement("Frozen Geode")], aliases=["田野调查收集包"], reward="Recycling Machine", room_reward="Friendship"),
    bundle("Fodder Bundle", "Bulletin Board", [requirement("Wheat", 10), requirement("Hay", 10), requirement("Apple", 3)], aliases=["饲料收集包"], reward="Heater", room_reward="Friendship"),
    bundle("Enchanter's Bundle", "Bulletin Board", [requirement("Oak Resin"), requirement("Wine"), requirement("Rabbit's Foot"), requirement("Pomegranate")], aliases=["魔法师收集包"], reward="Gold Bar (5)", room_reward="Friendship"),
    bundle("2,500 Bundle", "Vault", [requirement("Gold", 2500)], aliases=["2500金收集包"], reward="Chocolate Cake (3)", room_reward="Bus Repair"),
    bundle("5,000 Bundle", "Vault", [requirement("Gold", 5000)], aliases=["5000金收集包"], reward="Quality Fertilizer (30)", room_reward="Bus Repair"),
    bundle("10,000 Bundle", "Vault", [requirement("Gold", 10000)], aliases=["10000金收集包"], reward="Lightning Rod", room_reward="Bus Repair"),
    bundle("25,000 Bundle", "Vault", [requirement("Gold", 25000)], aliases=["25000金收集包"], reward="Crystalarium", room_reward="Bus Repair"),
]


def acquisition_record(entity: dict[str, Any]) -> dict[str, Any]:
    name = entity["name"]
    record_type = entity["record_type"]
    facts = entity["facts"]
    sources: list[dict[str, Any]] = []
    if record_type == "crop":
        sources.append({
            "source_type": "crop",
            "source_name": f"Harvest {name}",
            "location": "farm_or_valid_growing_area",
            "season": facts.get("seasons"),
            "weather": None,
            "time": None,
            "price": None,
            "probability": None,
            "quantity": (facts.get("harvest_quantity") or {}).get("minimum", 1),
            "conditions": [f"Grow {(facts.get('seed_name') or name)} for {facts.get('growth_days')} days."],
        })
        for source_name, price in sorted((facts.get("seed_buy_prices") or {}).items()):
            sources.append({
                "source_type": "festival" if "festival" in source_name else "shop",
                "source_name": source_name.replace("_", " ").title(),
                "location": None,
                "season": facts.get("seasons"),
                "weather": None,
                "time": None,
                "price": price,
                "probability": None,
                "quantity": 1,
                "conditions": [f"Purchase {facts.get('seed_name') or name}."],
            })
    elif record_type == "fish":
        for window in facts.get("availability_windows") or []:
            for location in window.get("locations") or []:
                sources.append({
                    "source_type": "fishing",
                    "source_name": location.replace("_", " ").title(),
                    "location": location,
                    "season": window.get("seasons"),
                    "weather": window.get("weather"),
                    "time": {"start": window.get("time_start"), "end": window.get("time_end")},
                    "price": None,
                    "probability": None,
                    "quantity": 1,
                    "conditions": list(facts.get("special_conditions") or []),
                })
    elif record_type == "recipe":
        sources.append({
            "source_type": facts.get("recipe_type"),
            "source_name": "Kitchen" if facts.get("recipe_type") == "cooking" else "Crafting Menu",
            "location": None,
            "season": None,
            "weather": None,
            "time": None,
            "price": None,
            "probability": None,
            "quantity": facts.get("result_quantity", 1),
            "conditions": [f"Know the recipe: {facts.get('unlock_source')}"],
        })
    return base_record(
        "acquisition",
        name,
        aliases=[],
        facts={"entity_name": name, "entity_type": record_type, "sources": sources},
        page_title=(entity.get("provenance") or {}).get("page_title") or name,
        section_title="Acquisition",
        record_id_suffix=f"{record_type}-{slug(name)}",
    )


SPECIAL_ACQUISITIONS = [
    ("Galaxy Sword", [{"source_type": "reward", "source_name": "Three Pillars", "location": "calico_desert", "season": None, "weather": None, "time": None, "price": None, "probability": None, "quantity": 1, "conditions": ["Hold a Prismatic Shard between the Three Pillars."]}]),
    ("Return Scepter", [{"source_type": "shop", "source_name": "Krobus", "location": "sewers", "season": None, "weather": None, "time": None, "price": 2000000, "probability": None, "quantity": 1, "conditions": []}]),
    ("Auto-Petter", [{"source_type": "shop", "source_name": "JojaMart", "location": "jojamart", "season": None, "weather": None, "time": None, "price": 50000, "probability": None, "quantity": 1, "conditions": ["Complete the Joja development route."]}, {"source_type": "reward", "source_name": "Skull Cavern treasure room", "location": "skull_cavern", "season": None, "weather": None, "time": None, "price": None, "probability": None, "quantity": 1, "conditions": ["Random treasure-room reward."]}]),
    ("Horse Flute", [{"source_type": "shop", "source_name": "Qi's Walnut Room", "location": "ginger_island", "season": None, "weather": None, "time": None, "price": 50, "currency": "Qi Gems", "probability": None, "quantity": 1, "conditions": []}]),
    ("Junimo Chest", [{"source_type": "shop", "source_name": "Qi's Walnut Room", "location": "ginger_island", "season": None, "weather": None, "time": None, "price": 30, "currency": "Qi Gems", "probability": None, "quantity": 2, "conditions": []}]),
    ("Deconstructor", [{"source_type": "shop", "source_name": "Qi's Walnut Room", "location": "ginger_island", "season": None, "weather": None, "time": None, "price": 20, "currency": "Qi Gems", "probability": None, "quantity": 1, "conditions": []}]),
    ("Training Rod", [{"source_type": "shop", "source_name": "Willy's Fish Shop", "location": "beach", "season": None, "weather": None, "time": None, "price": 25, "probability": None, "quantity": 1, "conditions": []}]),
    ("Fiberglass Rod", [{"source_type": "shop", "source_name": "Willy's Fish Shop", "location": "beach", "season": None, "weather": None, "time": None, "price": 1800, "probability": None, "quantity": 1, "conditions": ["Reach Fishing level 2."]}]),
    ("Iridium Rod", [{"source_type": "shop", "source_name": "Willy's Fish Shop", "location": "beach", "season": None, "weather": None, "time": None, "price": 7500, "probability": None, "quantity": 1, "conditions": ["Reach Fishing level 6."]}]),
    ("Milk Pail", [{"source_type": "shop", "source_name": "Marnie's Ranch", "location": "cindersap_forest", "season": None, "weather": None, "time": None, "price": 1000, "probability": None, "quantity": 1, "conditions": []}]),
    ("Shears", [{"source_type": "shop", "source_name": "Marnie's Ranch", "location": "cindersap_forest", "season": None, "weather": None, "time": None, "price": 1000, "probability": None, "quantity": 1, "conditions": []}]),
    ("Heater", [{"source_type": "shop", "source_name": "Marnie's Ranch", "location": "cindersap_forest", "season": None, "weather": None, "time": None, "price": 2000, "probability": None, "quantity": 1, "conditions": []}]),
    ("Auto-Grabber", [{"source_type": "shop", "source_name": "Marnie's Ranch", "location": "cindersap_forest", "season": None, "weather": None, "time": None, "price": 25000, "probability": None, "quantity": 1, "conditions": ["Reach Farming level 10."]}]),
    ("Backpack Upgrade", [{"source_type": "shop", "source_name": "Pierre's General Store", "location": "pelican_town", "season": None, "weather": None, "time": None, "price": 2000, "probability": None, "quantity": 1, "conditions": ["First inventory upgrade."]}, {"source_type": "shop", "source_name": "Pierre's General Store", "location": "pelican_town", "season": None, "weather": None, "time": None, "price": 10000, "probability": None, "quantity": 1, "conditions": ["Second inventory upgrade."]}]),
    ("Golden Clock", [{"source_type": "shop", "source_name": "Wizard's Magic Ink book", "location": "wizard_tower", "season": None, "weather": None, "time": None, "price": 10000000, "probability": None, "quantity": 1, "conditions": ["Return the Magic Ink and access the Wizard's building book."]}]),
]

ACQUISITIONS = [acquisition_record(item) for item in [*CROPS, *FISH, *RECIPES]]
for name, sources in SPECIAL_ACQUISITIONS:
    ACQUISITIONS.append(
        base_record(
            "acquisition",
            name,
            aliases=[],
            facts={"entity_name": name, "entity_type": "item", "sources": sources},
            page_title=name,
            section_title="Acquisition",
            record_id_suffix=f"item-{slug(name)}",
        )
    )

FACTS = [*CROPS, *FISH, *VILLAGERS, *RECIPES, *BUNDLES, *ACQUISITIONS]

GUIDE_PAGES: list[dict[str, Any]] = [
    {
        "title": "Getting Started",
        "intro": "Early-game progress is more reliable when the player keeps the first farm plot manageable, protects daily energy, and combines farming with exploration and income instead of trying to complete every system immediately.",
        "sections": {
            "First harvest": "Clear a small area, plant the starter Parsnip Seeds, and water planted crops every day. Crops do not advance on days when they are not watered, so a smaller plot that can be maintained consistently is safer than an oversized field.",
            "Early priorities": "Build a chest, forage while travelling, meet villagers, and use spare energy for fishing or farm cleanup. The Mines and Community Center become important medium-term objectives after their systems open.",
            "Planning discipline": "Keep enough time and energy to return home. Expand tools, crops, animals, and processing capacity gradually, and preserve cash for seeds, backpack space, and key upgrades.",
        },
    },
    {
        "title": "Day Cycle",
        "intro": "A Stardew Valley day has limited travel time and ends at night, so route planning is part of resource efficiency.",
        "sections": {
            "Time budget": "Group errands by location and avoid repeated cross-map travel. A task that is profitable in isolation may be inefficient when travel consumes the best part of the day.",
            "Late-night risk": "Returning before the character collapses avoids penalties and lost momentum. Leave dangerous areas with enough time to travel back.",
            "Daily reset": "Crops, machines, shops, weather, and NPC schedules change by day. Review the farm in the morning and choose a short list of priorities.",
        },
    },
    {
        "title": "Energy",
        "intro": "Energy constrains tool use, especially during the first weeks before upgrades and food become abundant.",
        "sections": {
            "Efficient tool use": "Clear only the land needed for the current plan. Avoid breaking every rock or chopping every tree when the resources are not immediately useful.",
            "Food": "Forage, inexpensive cooked items, and selected fish can extend work time. Reserve valuable ingredients when they are needed for bundles, gifts, or recipes.",
            "Recovery": "The Spa and later food options reduce energy pressure. Tool upgrades also reduce the number of actions needed for repetitive farm work.",
        },
    },
    {
        "title": "Seasons",
        "intro": "Most outdoor crops are seasonal and a normal season has 28 days, so planting dates must account for full growth time and any desired regrowth cycles.",
        "sections": {
            "Season transitions": "Most ordinary outdoor crops wither when their valid season ends. Multi-season crops can continue when the next season is also valid.",
            "Planting deadline": "A crop must finish all growth days by day 28 unless a valid growth-speed modifier is applied. Regrowing crops need additional time after the first harvest to realize their value.",
            "Winter": "Winter shifts emphasis toward mining, fishing, animals, processing, relationships, and the limited winter crops available in the current version.",
        },
    },
    {
        "title": "Weather",
        "intro": "Weather changes crop-watering workload, fish availability, lightning risk, and some schedules.",
        "sections": {
            "Rain": "Rain waters outdoor crops and enables rain-only fish. Use the saved watering time for mining, fishing, or errands.",
            "Storms": "Storms satisfy ordinary rain fishing conditions and can charge Lightning Rods. They also require attention to outdoor planning.",
            "Forecast": "Check the television forecast before committing to upgrades or fishing plans that depend on tomorrow's weather.",
        },
    },
    {
        "title": "Farming",
        "intro": "Farming combines crop scheduling, soil treatment, irrigation, animals, and machines into a production system.",
        "sections": {
            "Scale gradually": "Increase crop count only when watering capacity, sprinklers, and harvest labor can support it. Cash tied up in excessive seeds can delay key upgrades.",
            "Processing": "Kegs, Preserves Jars, and animal-product machines convert inputs into higher-value outputs but require materials, space, and time.",
            "Risk management": "Keep seeds, bundle items, and emergency cash separate from products intended for sale. Plan around seasonal deadlines.",
        },
    },
    {
        "title": "Crops",
        "intro": "Crop selection should consider season, growth days, regrowth, seed cost, harvest quantity, processing capacity, and the player's current goal.",
        "sections": {
            "Growth rules": "A crop advances one growth day when watered. The first harvest occurs after its full growth duration; a regrowing crop then uses its regrow interval.",
            "Trellis crops": "Trellis crops block walking through their tiles, so leave paths during layout planning.",
            "Decision criteria": "Choose crops differently for immediate cash, bundles, cooking, gifts, experience, or artisan processing. No single crop is best for every player state.",
        },
    },
    {
        "title": "Fishing",
        "intro": "Fishing success depends on the fish's season, weather, time window, location, and sometimes a special zone or event.",
        "sections": {
            "Match conditions": "Before travelling, check all availability conditions together. A correct season is not enough when weather, time, or location is wrong.",
            "Skill progression": "Early catches improve the fishing bar and unlock better rods, tackle, and recipes. The Training Rod can simplify the first levels.",
            "Income and collection": "Fishing can fund early upgrades while also supplying bundles, cooking ingredients, gifts, and fish-pond candidates.",
        },
    },
    {
        "title": "Fish",
        "intro": "The structured catalog stores separate availability windows so multiple combinations of season, weather, time, and location remain queryable.",
        "sections": {
            "Availability windows": "Some fish have multiple valid windows. The system should evaluate each window rather than merging conditions into an ambiguous sentence.",
            "Special locations": "The Mines, Desert, Secret Woods, Sewers, Ginger Island, Night Market, and legendary fishing zones have distinct rules.",
            "Planning": "Combine fish trips with nearby errands and keep difficult or seasonal fish on a checklist before the relevant season ends.",
        },
    },
    {
        "title": "Bundles",
        "intro": "Standard Community Center bundles contain fixed requirements, while Remixed Bundles use a different configuration and must never be silently substituted.",
        "sections": {
            "Rooms": "The Standard route contains six rooms and thirty bundles. Completing a room unlocks a larger restoration reward.",
            "Selection rules": "Some bundles require every listed item, while others require only a specified number from an optional group. Quality and quantity constraints matter.",
            "Version safety": "When Remixed data is incomplete, return a partial status or warning instead of presenting Standard requirements as Remixed requirements.",
        },
    },
    {
        "title": "Community Center",
        "intro": "The Community Center route turns farming, fishing, foraging, mining, animals, artisan production, and cash into a long-term collection plan.",
        "sections": {
            "Track seasonal items": "Prioritize items that disappear at the end of a season. Common materials and money bundles can be completed later.",
            "Reserve items": "Store one copy of likely bundle items before selling the rest. This reduces the cost of discovering a missed requirement late in the year.",
            "Route choice": "The Community Center and Joja routes are distinct. Guides should respect the route in player state rather than mixing goals.",
        },
    },
    {
        "title": "Friendship",
        "intro": "Friendship grows through conversation, gifts, quests, and events, with birthdays providing a strong multiplier for appropriate gifts.",
        "sections": {
            "Gift planning": "Use loved gifts when practical, but consider opportunity cost. A rare bundle item or essential crafting material may be more valuable elsewhere early in the game.",
            "Birthdays": "Record birthdays on a seasonal checklist and prepare gifts before the day arrives.",
            "Exceptions": "Universal gift rules have personal exceptions. A reliable knowledge base must keep individual preferences separate.",
        },
    },
    {
        "title": "Villagers",
        "intro": "Villager questions commonly involve birthdays, homes, marriage eligibility, schedules, gifts, and relationship progression.",
        "sections": {
            "Finding people": "Schedules vary with weather, season, events, and relationship state, so location answers may need context.",
            "Gift evidence": "The structured catalog exposes selected loved gifts and birthdays for deterministic answers.",
            "Natural queries": "English and Chinese aliases should resolve to the same canonical villager without asking the model to guess identity.",
        },
    },
    {
        "title": "Marriage",
        "intro": "Marriage requires relationship progression with an eligible candidate and additional game milestones.",
        "sections": {
            "Eligibility": "Only marriageable villagers follow the dating and marriage path. Other villagers still have friendship events and rewards.",
            "Preparation": "Build friendship consistently, view required events, and keep the relevant relationship item and home upgrade requirements in mind.",
            "After marriage": "Spouses move to the farmhouse and may help with selected chores, while friendship still requires attention.",
        },
    },
    {
        "title": "Cooking",
        "intro": "Cooking recipes combine structured ingredients with an unlock source such as television episodes, friendship, skills, shops, or locations.",
        "sections": {
            "Kitchen access": "Cooking requires a kitchen or equivalent access. Ingredient categories such as Any Fish should remain category substitutions rather than one specific item.",
            "Recipe collection": "Watch The Queen of Sauce, build relationships, raise skills, and check special shops to expand the recipe list.",
            "Use cases": "Cooked food supports energy, buffs, gifts, collections, and specific quests or bundles.",
        },
    },
    {
        "title": "Crafting",
        "intro": "Crafting converts materials into tools, machines, consumables, storage, and farm infrastructure.",
        "sections": {
            "Unlock sources": "Recipes can come from skill levels, quests, special orders, shops, events, or upgrades. The unlock source should be stated explicitly.",
            "Material planning": "Bars, Wood, Stone, Coal, Hardwood, and machine-specific materials often compete with tool and building upgrades.",
            "Automation": "Sprinklers and processing machines reduce repetitive labor or raise product value, but only when the farm can supply their inputs.",
        },
    },
    {
        "title": "The Mines",
        "intro": "The Mines provide ore, stone, geodes, combat drops, equipment, and progression through elevator checkpoints.",
        "sections": {
            "Preparation": "Bring food, manage inventory space, and choose whether the trip prioritizes depth, ore, monsters, or fishing.",
            "Elevator progress": "Reaching additional elevator checkpoints makes later resource runs more efficient.",
            "Risk": "Leave enough time and health to exit safely. A failed run can cost items and time.",
        },
    },
    {
        "title": "Combat",
        "intro": "Combat planning combines weapon reach, timing, food, defensive positioning, monster behavior, and the reason for entering a dangerous area.",
        "sections": {
            "Loadout": "Carry a suitable weapon, healing food, and enough free inventory slots. Rings, boots, and later upgrades change survivability.",
            "Positioning": "Avoid being surrounded and use terrain to limit approach angles. Retreat when health drops instead of risking the entire run.",
            "Objectives": "A monster-slaying goal, ore run, and fast descent use different routes and supplies.",
        },
    },
    {
        "title": "Skills",
        "intro": "Farming, Mining, Foraging, Fishing, and Combat unlock recipes and professions as experience increases.",
        "sections": {
            "Experience": "Experience is tied to relevant actions. Plan activities around desired recipe or profession unlocks.",
            "Professions": "Profession choices change economic or combat tradeoffs and should match the player's production plan.",
            "Recipe dependencies": "Many machines and utility items require a particular skill level, so skill progression affects infrastructure timing.",
        },
    },
    {
        "title": "Skull Cavern",
        "intro": "Skull Cavern rewards speed, preparation, luck, and resource investment more than slow completion of every floor.",
        "sections": {
            "Preparation": "Bring healing food, bombs, staircases, a strong weapon, and a plan for returning home. High-luck days improve the run's expected value.",
            "Descent strategy": "Use bombs on dense rock groups, take shafts when safe, and avoid spending too long fighting monsters unrelated to the goal.",
            "Tradeoffs": "Staircases and bombs cost resources, but can increase access to Iridium and treasure rooms. Measure a run by its objective, not only depth.",
        },
    },
    {
        "title": "Ginger Island",
        "intro": "Ginger Island adds Golden Walnut progression, island farming, the Volcano Dungeon, new crops, fishing areas, quests, and late-game systems.",
        "sections": {
            "Walnut progression": "Golden Walnuts unlock island facilities. Track solved and unsolved sources rather than searching randomly.",
            "Island farm": "The island farm supports crops year-round and changes the economics of multi-harvest and tropical crops.",
            "Volcano": "The Volcano Dungeon requires combat supplies and provides materials and access to advanced upgrades.",
        },
    },
    {
        "title": "Greenhouse",
        "intro": "The Greenhouse removes normal seasonal crop restrictions and provides a persistent layout for crops and fruit trees.",
        "sections": {
            "High-value use": "Regrowing crops and crops with long growth times benefit from persistent indoor seasons, but seed availability and processing capacity still matter.",
            "Layout": "Plan sprinklers, walkways, and fruit trees before filling the space. A stable layout reduces rework.",
            "Progression": "On the Community Center route, completing the Pantry restores the Greenhouse.",
        },
    },
    {
        "title": "Animals",
        "intro": "Animals require buildings, daily care, food, and processing capacity before they become reliable contributors to income and bundles.",
        "sections": {
            "Care": "Provide food, access, and winter heating where appropriate. Friendship and mood affect product quality.",
            "Processing": "Mayonnaise Machines, Cheese Presses, Looms, and Oil Makers increase the value of animal products.",
            "Expansion": "Add animals when hay supply, building space, and daily workload can support them.",
        },
    },
    {
        "title": "Artisan Goods",
        "intro": "Artisan production converts crops and animal products into higher-value goods over time.",
        "sections": {
            "Machine matching": "Use Kegs, Preserves Jars, Cheese Presses, Mayonnaise Machines, Looms, and Oil Makers for inputs suited to the player's time horizon.",
            "Bottlenecks": "Machine count, processing time, input supply, and storage can limit returns more than raw crop production.",
            "Planning": "Reserve bundle, gift, recipe, and seed-making inputs before processing everything for sale.",
        },
    },
    {
        "title": "Luck",
        "intro": "Daily luck influences several random systems and is especially relevant to mining and Skull Cavern planning.",
        "sections": {
            "Forecast": "Check the Fortune Teller when choosing between a high-variance mine run and deterministic farm work.",
            "Preparation": "Luck does not replace supplies. Bombs, staircases, food, weapons, and time management remain essential.",
            "Expected value": "Use better-luck days for activities where random outcomes have high value, while completing routine tasks on less favorable days.",
        },
    },
]


def guide_seed_rows() -> list[dict[str, Any]]:
    rows = []
    for index, page in enumerate(GUIDE_PAGES, start=1):
        body = [f"<p>{html.escape(page['intro'])}</p>"]
        for heading, paragraph in page["sections"].items():
            body.append(f"<h2>{html.escape(heading)}</h2>")
            body.append(f"<p>{html.escape(paragraph)}</p>")
        html_text = '<div class="mw-parser-output">\n' + "\n".join(body) + "\n</div>"
        digest = hashlib.sha256(html_text.encode("utf-8")).hexdigest()
        rows.append(
            {
                "schema_version": 1,
                "requested_title": page["title"],
                "title": page["title"],
                "page_id": None,
                "revision_id": None,
                "revision_timestamp": None,
                "source_url": wiki_url(page["title"]),
                "html": html_text,
                "sections_api": [],
                "categories": ["GameGuideLM offline seed"],
                "properties": [],
                "quality_status": "curated",
                "quality_flags": ["project_authored_summary"],
                "source_kind": "curated_seed",
                "retrieval_role": "guide",
                "discovery_priority": index,
                "language": "en",
                "source_name": "Official Stardew Valley Wiki (project-authored summary)",
                "license": {
                    "name": LICENSE_NAME,
                    "url": "https://creativecommons.org/licenses/by-nc-sa/3.0/",
                },
                "fetched_at": RETRIEVED_AT,
                "content_sha256": digest,
            }
        )
    return rows


def record_lookup(record_type: str, name: str) -> dict[str, Any] | None:
    key = normalize_name(name)
    for record in FACTS:
        if record["record_type"] == record_type and record["normalized_name"] == key:
            return record
    return None


def source_requirement(record_type: str, name: str) -> list[dict[str, Any]]:
    record = record_lookup(record_type, name)
    if record is None:
        return []
    source = record["provenance"]
    return [
        {
            "source_catalog_id": record["source_catalog_id"],
            "page_title": source["page_title"],
            "section_keywords": [source["section_title"]],
            "source_url": source["source_url"],
        }
    ]


def qa_spec(
    category: str,
    intent: str,
    status: str,
    en_question: str,
    zh_question: str,
    *,
    entity: str | None = None,
    player_state: dict[str, Any] | None = None,
    must_include: Iterable[str] = (),
    must_not_include: Iterable[str] = (),
    record_type: str | None = None,
    reference_answer: str = "",
    difficulty: str = "medium",
) -> dict[str, Any]:
    return {
        "category": category,
        "intent": intent,
        "expected_status": status,
        "en_question": en_question,
        "zh_question": zh_question,
        "entity": entity,
        "player_state": player_state or {},
        "must_include": list(must_include),
        "must_not_include": list(must_not_include),
        "record_type": record_type,
        "reference_answer": reference_answer,
        "difficulty": difficulty,
    }


EVALUATION_SPECS: list[dict[str, Any]] = [
    # Crop / seasonal planning: 13 found, 4 needs_context, 3 not_found.
    qa_spec("crop", "crop_info", "found", "How long does Parsnip take to grow?", "Parsnip需要几天成熟？", entity="Parsnip", must_include=["4"], record_type="crop", reference_answer="Parsnip takes 4 days to mature."),
    qa_spec("crop", "crop_info", "found", "Does Strawberry regrow after harvest?", "Strawberry收获后会再生吗？", entity="Strawberry", must_include=["4"], record_type="crop", reference_answer="Strawberry regrows every 4 days after the first harvest."),
    qa_spec("crop", "crop_info", "found", "Which season can Blueberry grow in?", "Blueberry在哪个季节种植？", entity="Blueberry", must_include=["Summer"], record_type="crop", reference_answer="Blueberry is a Summer crop."),
    qa_spec("crop", "crop_info", "found", "Can Corn grow across two seasons?", "Corn可以跨两个季节生长吗？", entity="Corn", must_include=["Summer", "Fall"], record_type="crop", reference_answer="Corn grows in Summer and Fall."),
    qa_spec("crop", "crop_info", "found", "How often do Cranberries regrow?", "Cranberries多久再生一次？", entity="Cranberries", must_include=["5"], record_type="crop", reference_answer="Cranberries regrow every 5 days."),
    qa_spec("crop", "crop_info", "found", "How many days does Pumpkin take to mature?", "Pumpkin成熟需要多少天？", entity="Pumpkin", must_include=["13"], record_type="crop", reference_answer="Pumpkin takes 13 days."),
    qa_spec("crop", "crop_info", "found", "Can Pineapple grow year-round on Ginger Island?", "Pineapple在姜岛能全年种吗？", entity="Pineapple", must_include=["Summer", "Ginger Island", "all seasons"], record_type="crop", reference_answer="Pineapple grows in Summer in the Valley and in all seasons on Ginger Island."),
    qa_spec("crop", "crop_info", "found", "What is the growth and regrowth time for Ancient Fruit?", "Ancient Fruit的成熟和再生时间是多少？", entity="Ancient Fruit", must_include=["28", "7"], record_type="crop", reference_answer="Ancient Fruit takes 28 days, then regrows every 7 days."),
    qa_spec("crop", "crop_deadline", "found", "Can I harvest Cauliflower if planted on Spring day 16?", "春季第16天种Cauliflower还能收获吗？", entity="Cauliflower", player_state={"season": "spring", "day": 16}, must_include=["28"], record_type="crop", reference_answer="It matures on Spring 28."),
    qa_spec("crop", "crop_deadline", "found", "Can I harvest Yam if planted on Fall day 18?", "秋季第18天种Yam还能收获吗？", entity="Yam", player_state={"season": "fall", "day": 18}, must_include=["28"], record_type="crop", reference_answer="It matures on Fall 28."),
    qa_spec("crop", "crop_deadline", "found", "Can I harvest Melon if planted on Summer day 16?", "夏季第16天种Melon还能收获吗？", entity="Melon", player_state={"season": "summer", "day": 16}, must_include=["28"], record_type="crop", reference_answer="It matures on Summer 28."),
    qa_spec("crop", "crop_deadline", "found", "Can I harvest Pumpkin if planted on Fall day 15?", "秋季第15天种Pumpkin还能收获吗？", entity="Pumpkin", player_state={"season": "fall", "day": 15}, must_include=["28"], record_type="crop", reference_answer="It matures on Fall 28."),
    qa_spec("crop", "crop_deadline", "found", "Can I harvest Potato if planted on Spring day 22?", "春季第22天种Potato还能收获吗？", entity="Potato", player_state={"season": "spring", "day": 22}, must_include=["28"], record_type="crop", reference_answer="It matures on Spring 28."),
    qa_spec("crop", "crop_deadline", "needs_context", "Can I still plant Cauliflower in time?", "Cauliflower还来得及种吗？", entity="Cauliflower", record_type="crop", reference_answer="Season and day are required."),
    qa_spec("crop", "crop_deadline", "needs_context", "Can I still plant Starfruit in time?", "Starfruit现在种还来得及吗？", entity="Starfruit", record_type="crop", reference_answer="Season and day are required."),
    qa_spec("crop", "crop_deadline", "needs_context", "Can I still plant Pumpkin before the season ends?", "Pumpkin在季末前还来得及种吗？", entity="Pumpkin", record_type="crop", reference_answer="Season and day are required."),
    qa_spec("crop", "crop_deadline", "needs_context", "Will Sweet Gem Berry mature if I plant it now?", "Sweet Gem Berry现在种能成熟吗？", entity="Sweet Gem Berry", record_type="crop", reference_answer="Season and day are required."),
    qa_spec("crop", "crop_info", "not_found", "How long does Void Melon take to grow?", "Void Melon需要几天成熟？", entity="Void Melon", reference_answer="No such crop exists in the tracked snapshot."),
    qa_spec("crop", "crop_info", "not_found", "How many days does Moonberry take to mature?", "Moonberry几天成熟？", entity="Moonberry", reference_answer="No such crop exists in the tracked snapshot."),
    qa_spec("crop", "crop_info", "not_found", "How long does Galaxy Pumpkin take to grow?", "Galaxy Pumpkin需要几天成熟？", entity="Galaxy Pumpkin", reference_answer="No such crop exists in the tracked snapshot."),

    # Fish: 9 found, 4 needs_context, 2 not_found.
    qa_spec("fish", "fish_availability", "found", "Where and when can I catch Catfish?", "Catfish在哪里、什么时候能钓？", entity="Catfish", must_include=["River", "Rain"], record_type="fish", reference_answer="Catfish has rain-based river and special-location windows."),
    qa_spec("fish", "fish_availability", "found", "When can I catch Eel?", "Eel什么时候能钓？", entity="Eel", must_include=["4:00 PM", "2:00 AM"], record_type="fish", reference_answer="Eel is an evening and night rain fish in Spring and Fall."),
    qa_spec("fish", "fish_availability", "found", "What conditions are required for Pufferfish?", "Pufferfish需要什么天气和时间？", entity="Pufferfish", must_include=["Summer", "Sunny"], record_type="fish", reference_answer="Pufferfish requires Summer sun from noon to 4 PM at the ocean."),
    qa_spec("fish", "fish_availability", "found", "Where can I catch Sturgeon?", "Sturgeon在哪里钓？", entity="Sturgeon", must_include=["Mountain Lake"], record_type="fish", reference_answer="Sturgeon is caught in the mountain lake in Summer and Winter."),
    qa_spec("fish", "fish_availability", "found", "What weather does Walleye require?", "Walleye需要什么天气？", entity="Walleye", must_include=["Rain"], record_type="fish", reference_answer="Walleye requires rain in Fall or Rain Totem conditions in Winter."),
    qa_spec("fish", "fish_availability", "found", "Where can I catch Lava Eel?", "Lava Eel在哪里钓？", entity="Lava Eel", must_include=["Mines 100", "Volcano Caldera"], record_type="fish", reference_answer="Lava Eel is available on Mines floor 100 and at the Volcano Caldera."),
    qa_spec("fish", "fish_availability", "found", "Where is Stingray available?", "Stingray在哪里能钓？", entity="Stingray", must_include=["Pirate Cove"], record_type="fish", reference_answer="Stingray is caught at Pirate Cove."),
    qa_spec("fish", "fish_availability", "found", "When and where can I catch Legend?", "Legend在哪里、什么时候能钓？", entity="Legend", must_include=["Spring", "Rain", "Mountain Lake"], record_type="fish", reference_answer="Legend is a Spring rain fish in the mountain lake special zone."),
    qa_spec("fish", "fish_availability", "found", "When can Midnight Squid be caught?", "Midnight Squid什么时候能钓？", entity="Midnight Squid", must_include=["Winter", "Night Market Submarine"], record_type="fish", reference_answer="It is available during the Night Market submarine window."),
    qa_spec("fish", "fish_availability", "needs_context", "Can I catch Catfish right now?", "我现在能钓到Catfish吗？", entity="Catfish", record_type="fish", reference_answer="Season, weather, time, and location are required."),
    qa_spec("fish", "fish_availability", "needs_context", "Can I catch Eel today?", "我今天能钓Eel吗？", entity="Eel", record_type="fish", reference_answer="Season, weather, time, and location are required."),
    qa_spec("fish", "fish_availability", "needs_context", "Is Pufferfish catchable now?", "Pufferfish现在能钓吗？", entity="Pufferfish", record_type="fish", reference_answer="Season, weather, time, and location are required."),
    qa_spec("fish", "fish_availability", "needs_context", "Can I catch Walleye right now?", "Walleye现在能钓吗？", entity="Walleye", record_type="fish", reference_answer="Season, weather, time, and location are required."),
    qa_spec("fish", "fish_availability", "not_found", "Where can I catch Galaxy Catfish?", "Galaxy Catfish在哪里钓？", entity="Galaxy Catfish", reference_answer="No such fish exists in the tracked snapshot."),
    qa_spec("fish", "fish_availability", "not_found", "When can I catch Moon Eel?", "Moon Eel什么时候能钓？", entity="Moon Eel", reference_answer="No such fish exists in the tracked snapshot."),

    # Villagers: 13 found, 2 not_found.
    qa_spec("villager", "villager_gifts", "found", "What gifts does Abigail love?", "阿比盖尔喜欢什么礼物？", entity="Abigail", must_include=["Amethyst"], record_type="villager", reference_answer="Abigail loves gifts including Amethyst."),
    qa_spec("villager", "villager_gifts", "found", "What gifts does Alex love?", "亚历克斯喜欢什么礼物？", entity="Alex", must_include=["Complete Breakfast"], record_type="villager", reference_answer="Alex loves Complete Breakfast and Salmon Dinner."),
    qa_spec("villager", "villager_gifts", "found", "What does Elliott love?", "艾利欧特最爱什么？", entity="Elliott", must_include=["Crab Cakes"], record_type="villager", reference_answer="Elliott loves Crab Cakes among other gifts."),
    qa_spec("villager", "villager_gifts", "found", "What gifts does Emily love?", "艾米丽喜欢什么礼物？", entity="Emily", must_include=["Amethyst"], record_type="villager", reference_answer="Emily loves gems and selected artisan items."),
    qa_spec("villager", "villager_gifts", "found", "What gifts does Haley love?", "海莉喜欢什么礼物？", entity="Haley", must_include=["Coconut"], record_type="villager", reference_answer="Haley loves Coconut, Fruit Salad, Pink Cake, and Sunflower."),
    qa_spec("villager", "villager_gifts", "found", "What does Harvey love?", "哈维最爱什么礼物？", entity="Harvey", must_include=["Coffee"], record_type="villager", reference_answer="Harvey loves Coffee and selected artisan foods."),
    qa_spec("villager", "villager_gifts", "found", "What gifts does Leah love?", "莉亚喜欢什么礼物？", entity="Leah", must_include=["Goat Cheese"], record_type="villager", reference_answer="Leah loves Goat Cheese among other gifts."),
    qa_spec("villager", "villager_gifts", "found", "What gifts does Maru love?", "玛鲁喜欢什么礼物？", entity="Maru", must_include=["Battery Pack"], record_type="villager", reference_answer="Maru loves Battery Pack and several crafted or high-value items."),
    qa_spec("villager", "villager_gifts", "found", "What does Penny love?", "潘妮最爱什么？", entity="Penny", must_include=["Melon"], record_type="villager", reference_answer="Penny loves Melon among other gifts."),
    qa_spec("villager", "villager_gifts", "found", "What gifts does Sebastian love?", "塞巴斯蒂安喜欢什么礼物？", entity="Sebastian", must_include=["Frozen Tear"], record_type="villager", reference_answer="Sebastian loves Frozen Tear and selected dark-themed items."),
    qa_spec("villager", "villager_gifts", "found", "What gifts does Shane love?", "谢恩喜欢什么礼物？", entity="Shane", must_include=["Hot Pepper"], record_type="villager", reference_answer="Shane loves Hot Pepper among other gifts."),
    qa_spec("villager", "villager_gifts", "found", "What gifts does Willy love?", "威利喜欢什么礼物？", entity="Willy", must_include=["Catfish"], record_type="villager", reference_answer="Willy loves several fish and high-value items."),
    qa_spec("villager", "villager_gifts", "found", "What gifts does Krobus love?", "科罗布斯喜欢什么礼物？", entity="Krobus", must_include=["Void Egg"], record_type="villager", reference_answer="Krobus loves Void Egg among other gifts."),
    qa_spec("villager", "villager_gifts", "not_found", "What gifts does Gunther love?", "Gunther喜欢什么礼物？", entity="Gunther", reference_answer="No tracked gift profile is available for Gunther."),
    qa_spec("villager", "villager_gifts", "not_found", "What gifts does Morris love?", "Morris喜欢什么礼物？", entity="Morris", reference_answer="No tracked gift profile is available for Morris."),

    # Recipes: 13 found, 2 not_found.
    qa_spec("recipe", "recipe", "found", "How do I craft Quality Sprinkler?", "Quality Sprinkler怎么制作？", entity="Quality Sprinkler", must_include=["Iron Bar", "Gold Bar", "Refined Quartz"], record_type="recipe", reference_answer="It uses one Iron Bar, Gold Bar, and Refined Quartz."),
    qa_spec("recipe", "recipe", "found", "How do I craft Keg?", "Keg怎么制作？", entity="Keg", must_include=["Oak Resin"], record_type="recipe", reference_answer="Keg requires Wood, Copper Bar, Iron Bar, and Oak Resin."),
    qa_spec("recipe", "recipe", "found", "How do I craft Fish Smoker?", "Fish Smoker怎么制作？", entity="Fish Smoker", must_include=["Hardwood", "Sea Jelly"], record_type="recipe", reference_answer="Fish Smoker uses Hardwood and three jelly types."),
    qa_spec("recipe", "recipe", "found", "How do I make Sashimi?", "Sashimi怎么做？", entity="Sashimi", must_include=["Any Fish"], record_type="recipe", reference_answer="Sashimi uses one Any Fish category ingredient."),
    qa_spec("recipe", "recipe", "found", "How do I cook Pumpkin Soup?", "Pumpkin Soup怎么烹饪？", entity="Pumpkin Soup", must_include=["Pumpkin", "Milk"], record_type="recipe", reference_answer="Pumpkin Soup uses Pumpkin and Milk."),
    qa_spec("recipe", "recipe", "found", "What is the recipe for Triple Shot Espresso?", "Triple Shot Espresso的配方是什么？", entity="Triple Shot Espresso", must_include=["Coffee", "3"], record_type="recipe", reference_answer="It uses three Coffee."),
    qa_spec("recipe", "recipe", "found", "How do I craft Iridium Sprinkler?", "Iridium Sprinkler怎么制作？", entity="Iridium Sprinkler", must_include=["Iridium Bar", "Battery Pack"], record_type="recipe", reference_answer="It uses Gold Bar, Iridium Bar, and Battery Pack."),
    qa_spec("recipe", "recipe", "found", "How do I make Crab Cakes?", "Crab Cakes怎么做？", entity="Crab Cakes", must_include=["Crab", "Wheat Flour", "Oil"], record_type="recipe", reference_answer="Crab Cakes use Crab, Wheat Flour, Egg, and Oil."),
    qa_spec("recipe", "recipe", "found", "How do I cook Tropical Curry?", "Tropical Curry怎么做？", entity="Tropical Curry", must_include=["Coconut", "Pineapple", "Hot Pepper"], record_type="recipe", reference_answer="It uses Coconut, Pineapple, and Hot Pepper."),
    qa_spec("recipe", "recipe", "found", "How do I craft Crystalarium?", "Crystalarium怎么制作？", entity="Crystalarium", must_include=["Stone", "Iridium Bar", "Battery Pack"], record_type="recipe", reference_answer="Crystalarium uses Stone, Gold Bars, Iridium Bars, and a Battery Pack."),
    qa_spec("recipe", "recipe", "found", "How do I make Banana Pudding?", "Banana Pudding怎么做？", entity="Banana Pudding", must_include=["Banana", "Milk", "Sugar"], record_type="recipe", reference_answer="Banana Pudding uses Banana, Milk, and Sugar."),
    qa_spec("recipe", "recipe", "found", "How do I craft Seed Maker?", "Seed Maker怎么制作？", entity="Seed Maker", must_include=["Wood", "Coal", "Gold Bar"], record_type="recipe", reference_answer="Seed Maker uses Wood, Coal, and a Gold Bar."),
    qa_spec("recipe", "recipe", "found", "How do I cook Lucky Lunch?", "Lucky Lunch怎么做？", entity="Lucky Lunch", must_include=["Sea Cucumber", "Tortilla", "Blue Jazz"], record_type="recipe", reference_answer="Lucky Lunch uses Sea Cucumber, Tortilla, and Blue Jazz."),
    qa_spec("recipe", "recipe", "not_found", "How do I craft Golden Catfish Sword?", "Golden Catfish Sword怎么制作？", entity="Golden Catfish Sword", reference_answer="No such recipe is tracked."),
    qa_spec("recipe", "recipe", "not_found", "What is the recipe for Iridium Keg?", "Iridium Keg的配方是什么？", entity="Iridium Keg", reference_answer="No such recipe is tracked."),

    # Bundles: 5 found, 10 partial for remixed mode.
    qa_spec("bundle", "bundle", "found", "What does the River Fish Bundle require?", "River Fish Bundle需要什么？", entity="River Fish Bundle", must_include=["Catfish", "Shad"], record_type="bundle", reference_answer="The Standard River Fish Bundle requires Sunfish, Catfish, Shad, and Tiger Trout."),
    qa_spec("bundle", "bundle", "found", "What does the Quality Crops Bundle require?", "Quality Crops Bundle需要什么？", entity="Quality Crops Bundle", must_include=["Parsnip", "Melon", "Pumpkin", "Corn"], record_type="bundle", reference_answer="Choose three of four gold-quality crop stacks."),
    qa_spec("bundle", "bundle", "found", "What is in the Artisan Bundle?", "Artisan Bundle里需要哪些物品？", entity="Artisan Bundle", must_include=["Truffle Oil", "Cloth", "Cheese"], record_type="bundle", reference_answer="Choose six of twelve artisan or fruit items."),
    qa_spec("bundle", "bundle", "found", "What does the Chef's Bundle require?", "Chef's Bundle需要什么？", entity="Chef's Bundle", must_include=["Maple Syrup", "Truffle", "Fried Egg"], record_type="bundle", reference_answer="The Chef's Bundle has six fixed requirements."),
    qa_spec("bundle", "bundle", "found", "How much gold is required for the 25,000 Bundle?", "25,000 Bundle需要多少金币？", entity="25,000 Bundle", must_include=["25000"], record_type="bundle", reference_answer="It requires 25,000 gold."),
    *[
        qa_spec(
            "bundle", "bundle", "partial",
            f"What does the Remixed {name} require?",
            f"混合收集包里的{name}需要什么？",
            entity=name,
            player_state={"bundle_mode": "remixed"},
            record_type="bundle",
            reference_answer="The curated release does not claim complete Remixed Bundle coverage.",
            difficulty="hard",
        )
        for name in [
            "Spring Crops Bundle", "Summer Crops Bundle", "Fall Crops Bundle",
            "River Fish Bundle", "Lake Fish Bundle", "Ocean Fish Bundle",
            "Chef's Bundle", "Dye Bundle", "Animal Bundle", "Artisan Bundle",
        ]
    ],

    # Acquisition: 9 found, 1 not_found.
    qa_spec("acquisition", "acquisition", "found", "Where can I get Strawberry?", "Strawberry怎么获得？", entity="Strawberry", must_include=["Egg Festival"], record_type="acquisition", reference_answer="Grow Strawberry or buy Strawberry Seeds at the Egg Festival."),
    qa_spec("acquisition", "acquisition", "found", "How do I get Galaxy Sword?", "Galaxy Sword怎么获得？", entity="Galaxy Sword", must_include=["Three Pillars", "Prismatic Shard"], record_type="acquisition", reference_answer="Hold a Prismatic Shard between the Three Pillars."),
    qa_spec("acquisition", "acquisition", "found", "Where can I buy Return Scepter?", "Return Scepter在哪里买？", entity="Return Scepter", must_include=["Krobus", "2000000"], record_type="acquisition", reference_answer="Krobus sells it for 2,000,000g."),
    qa_spec("acquisition", "acquisition", "found", "How can I get Auto-Petter?", "Auto-Petter怎么获得？", entity="Auto-Petter", must_include=["JojaMart", "Skull Cavern"], record_type="acquisition", reference_answer="It can be bought on the Joja route or found in Skull Cavern treasure rooms."),
    qa_spec("acquisition", "acquisition", "found", "Where can I buy Iridium Rod?", "Iridium Rod在哪里买？", entity="Iridium Rod", must_include=["Willy", "7500"], record_type="acquisition", reference_answer="Willy sells it after Fishing level 6."),
    qa_spec("acquisition", "acquisition", "found", "How do I get Backpack Upgrade?", "Backpack Upgrade怎么获得？", entity="Backpack Upgrade", must_include=["Pierre", "2000", "10000"], record_type="acquisition", reference_answer="Pierre sells two inventory upgrades."),
    qa_spec("acquisition", "acquisition", "found", "How can I obtain Fish Smoker?", "Fish Smoker如何获得？", entity="Fish Smoker", must_include=["Crafting Menu"], record_type="acquisition", reference_answer="Craft it after learning the recipe."),
    qa_spec("acquisition", "acquisition", "found", "Where can I get Catfish?", "Catfish怎么获得？", entity="Catfish", must_include=["River"], record_type="acquisition", reference_answer="Catch it by fishing in its valid locations and conditions."),
    qa_spec("acquisition", "acquisition", "found", "How can I get Quality Sprinkler?", "Quality Sprinkler怎么获得？", entity="Quality Sprinkler", must_include=["Crafting Menu"], record_type="acquisition", reference_answer="Craft it after reaching the required Farming level."),
    qa_spec("acquisition", "acquisition", "not_found", "Where can I get Dragon Tractor?", "Dragon Tractor怎么获得？", entity="Dragon Tractor", reference_answer="No acquisition source is tracked for that nonexistent item."),

    # Guide / progression: 8 found, 2 needs_context.
    qa_spec("guide", "guide", "found", "What should I prioritize during the first spring?", "第一年春季应该优先做什么？", must_include=["Getting Started"], record_type=None, reference_answer="Keep the first farm manageable and balance crops, storage, exploration, and income."),
    qa_spec("guide", "guide", "found", "How should I prepare for Skull Cavern?", "沙漠矿洞应该怎么准备？", must_include=["Skull Cavern"], record_type=None, reference_answer="Bring food, bombs, staircases, a strong weapon, and use favorable luck."),
    qa_spec("guide", "guide", "found", "How should I use the Greenhouse?", "温室应该怎么规划？", must_include=["Greenhouse"], record_type=None, reference_answer="Use persistent layouts for long-growth and regrowing crops."),
    qa_spec("guide", "guide", "found", "How can I improve friendship efficiently?", "怎么高效提升村民好感？", must_include=["Friendship"], record_type=None, reference_answer="Talk, plan gifts and birthdays, and respect personal exceptions."),
    qa_spec("guide", "guide", "found", "How do I unlock more cooking recipes?", "怎么解锁更多烹饪配方？", must_include=["Cooking"], record_type=None, reference_answer="Use television, friendship, skills, shops, and special locations."),
    qa_spec("guide", "guide", "found", "What should I focus on first on Ginger Island?", "刚到姜岛应该先做什么？", must_include=["Ginger Island"], record_type=None, reference_answer="Track Golden Walnuts and unlock facilities deliberately."),
    qa_spec("guide", "guide", "found", "When should I expand into animals?", "什么时候适合开始养动物？", must_include=["Animals"], record_type=None, reference_answer="Expand when buildings, feed, processing, and daily workload are sustainable."),
    qa_spec("guide", "guide", "found", "How should I plan Community Center bundles?", "社区中心收集包应该怎么规划？", must_include=["Community Center"], record_type=None, reference_answer="Track seasonal items, reserve one copy, and respect route and bundle mode."),
    qa_spec("guide", "guide", "needs_context", "What should I plant today?", "我今天应该种什么？", reference_answer="Season and day are required for a reliable recommendation."),
    qa_spec("guide", "guide", "needs_context", "Which crop should I plant right now?", "我现在应该种哪种作物？", reference_answer="Season and day are required for a reliable recommendation."),
]


ZH_REQUIREMENT_EQUIVALENTS: dict[str, str] = {
    "Spring": "春季",
    "Summer": "夏季",
    "Fall": "秋季",
    "Winter": "冬季",
    "Rain": "雨天",
    "4:00 PM": "下午 4:00",
    "2:00 AM": "次日凌晨 2:00",
    "Mines 100": "矿井100层",
    "Volcano Caldera": "火山口",
    "Pirate Cove": "海盗湾",
    "Mountain Lake": "山地湖泊",
    "Night Market Submarine": "夜市潜水艇",
    "Oak Resin": "橡树树脂",
    "Wood": "木材",
    "Coal": "煤炭",
    "Stone": "石头",
    "River": "河流",
    "Ginger Island": "姜岛",
    "all seasons": "全年",
}


def language_requirements(values: Iterable[str], language: str) -> list[Any]:
    result: list[Any] = []
    for value in values:
        text = str(value)
        translated = ZH_REQUIREMENT_EQUIVALENTS.get(text) if language == "zh" else None
        result.append({"any_of": [text, translated]} if translated else text)
    return result


def evaluation_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(EVALUATION_SPECS) != 100:
        raise AssertionError(f"Expected 100 evaluation specs, found {len(EVALUATION_SPECS)}")
    validation: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    split_counters = {"validation": 0, "eval": 0}
    for index, spec in enumerate(EVALUATION_SPECS):
        language = "en" if index % 2 == 0 else "zh"
        split = "validation" if index % 5 in {0, 1} else "eval"
        split_counters[split] += 1
        entity = spec.get("entity")
        source_type = spec.get("record_type")
        required_sources = source_requirement(source_type, entity) if source_type and entity else []
        record = {
            "schema_version": 1,
            "id": f"stardew_{split}_{split_counters[split]:03d}",
            "game": "stardew_valley",
            "game_version": GAME_VERSION,
            "platform": "all",
            "language": language,
            "question": spec[f"{language}_question"],
            "intent": spec["intent"],
            "entities": [entity] if entity else [],
            "player_state": spec.get("player_state") or {},
            "expected_status": spec["expected_status"],
            "required_facts": language_requirements(spec.get("must_include") or [], language),
            "required_sources": required_sources,
            "reference_answer": spec.get("reference_answer") or "",
            "must_include": language_requirements(spec.get("must_include") or [], language),
            "must_not_include": list(spec.get("must_not_include") or []),
            "difficulty": spec.get("difficulty") or "medium",
            "split": split,
            "category": spec["category"],
            "annotator": "deterministic_release_generator_v1",
            "reviewer": None,
            "review_status": "machine_validated",
            "human_review_required": True,
            "review_notes": "Requires independent human source review before being marked approved.",
        }
        (validation if split == "validation" else evaluation).append(record)
    return validation, evaluation


def grounded_training_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for crop_record in CROPS:
        facts = crop_record["facts"]
        evidence = (
            f"[S1] {crop_record['name']}: seasons={','.join(facts['seasons'])}; "
            f"growth_days={facts['growth_days']}; regrow_days={facts.get('regrow_days')}; "
            f"base_sell_price={facts['base_sell_price']}g."
        )
        answer = (
            f"{crop_record['name']} grows in {', '.join(season.title() for season in facts['seasons'])} "
            f"and takes {facts['growth_days']} days to mature. [S1]"
        )
        records.append({
            "game": "stardew_valley", "language": "en", "intent": "crop_info",
            "question": f"Summarize the growing conditions for {crop_record['name']}.",
            "evidence": evidence, "answer": answer, "source_url": crop_record["provenance"]["source_url"],
        })
    for fish_record in FISH:
        window = fish_record["facts"]["availability_windows"][0]
        evidence = (
            f"[S1] {fish_record['name']}: seasons={','.join(window['seasons'])}; "
            f"weather={','.join(window['weather'])}; time={window['time_start']}-{window['time_end']}; "
            f"locations={','.join(window['locations'])}."
        )
        answer = (
            f"{fish_record['name']} is available in {', '.join(window['seasons'])}, "
            f"under {', '.join(window['weather'])} weather, from {window['time_start']} to {window['time_end']} "
            f"at {', '.join(window['locations'])}. [S1]"
        )
        records.append({
            "game": "stardew_valley", "language": "en", "intent": "fish_availability",
            "question": f"Use the evidence to state the first tracked availability window for {fish_record['name']}.",
            "evidence": evidence, "answer": answer, "source_url": fish_record["provenance"]["source_url"],
        })
    for recipe_record in RECIPES[:80]:
        facts = recipe_record["facts"]
        ingredient_text = ", ".join(f"{item['item_name']} x{item['quantity']}" for item in facts["ingredients"])
        evidence = f"[S1] {recipe_record['name']} ({facts['recipe_type']}): {ingredient_text}. Unlock: {facts['unlock_source']}."
        answer = f"{recipe_record['name']} requires {ingredient_text}. Unlock source: {facts['unlock_source']}. [S1]"
        records.append({
            "game": "stardew_valley", "language": "en", "intent": "recipe",
            "question": f"What ingredients and unlock source are recorded for {recipe_record['name']}?",
            "evidence": evidence, "answer": answer, "source_url": recipe_record["provenance"]["source_url"],
        })
    output: list[dict[str, Any]] = []
    for index, item in enumerate(records):
        split = "validation" if index % 10 == 9 else "train"
        prompt = (
            "Answer only from the evidence. Cite [S1]. If the evidence is insufficient, say so.\n\n"
            f"EVIDENCE:\n{item['evidence']}\n\nQUESTION:\n{item['question']}"
        )
        output.append({
            "id": f"stardew_grounded_{split}_{index + 1:04d}",
            "split": split,
            "game": "stardew_valley",
            "domain": "stardew_valley",
            "language": item["language"],
            "intent": item["intent"],
            "messages": [
                {"role": "system", "content": "You are a grounded Stardew Valley guide assistant."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": item["answer"]},
            ],
            "required_facts": [item["answer"].split(". [S1]")[0]],
            "forbidden_errors": ["unsupported fact"],
            "source_urls": [item["source_url"]],
            "verified": False,
            "machine_validated": True,
            "verification_method": "deterministic_render_from_structured_catalog",
            "review_status": "machine_validated",
            "generation_method": "deterministic_template",
            "dataset_version": "stardew_grounded_v1",
        })
    return [row for row in output if row["split"] == "train"], [row for row in output if row["split"] == "validation"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_release_data() -> dict[str, Any]:
    catalog_root = PROJECT_ROOT / "data" / "stardew" / "catalog"
    cleaned_root = catalog_root / "cleaned"
    reports_root = catalog_root / "reports"
    evaluation_root = PROJECT_ROOT / "data" / "stardew" / "evaluation"
    training_root = PROJECT_ROOT / "data" / "stardew" / "training"
    guide_seed_path = PROJECT_ROOT / "data" / "stardew" / "guides" / "seed" / "pages.jsonl"
    for directory in (cleaned_root, reports_root, evaluation_root, training_root, guide_seed_path.parent):
        directory.mkdir(parents=True, exist_ok=True)

    ordered_facts = sorted(FACTS, key=lambda row: (row["record_type"], row["normalized_name"], row["source_catalog_id"]))
    ids = [row["source_catalog_id"] for row in ordered_facts]
    if len(ids) != len(set(ids)):
        duplicates = [item for item, count in Counter(ids).items() if count > 1]
        raise AssertionError(f"Duplicate generated IDs: {duplicates[:10]}")

    facts_path = cleaned_root / "facts.jsonl"
    write_jsonl(facts_path, ordered_facts)
    type_file_names = {
        "crop": "crops.jsonl",
        "fish": "fish.jsonl",
        "villager": "villagers.jsonl",
        "recipe": "recipes.jsonl",
        "bundle": "bundles.jsonl",
        "acquisition": "acquisition_sources.jsonl",
    }
    for record_type, file_name in type_file_names.items():
        write_jsonl(
            cleaned_root / file_name,
            [row for row in ordered_facts if row["record_type"] == record_type],
        )

    type_counts = Counter(row["record_type"] for row in ordered_facts)
    acquisition_relations = sum(
        len((row.get("facts") or {}).get("sources") or [])
        for row in ordered_facts
        if row["record_type"] == "acquisition"
    )
    snapshot_manifest = {
        "schema_version": 1,
        "game": "stardew_valley",
        "game_version": GAME_VERSION,
        "platform": "all",
        "record_count": len(ordered_facts),
        "record_type_counts": dict(sorted(type_counts.items())),
        "acquisition_relation_count": acquisition_relations,
        "source_page_count": len({row["provenance"]["page_title"] for row in ordered_facts}),
        "source_name": "Official Stardew Valley Wiki",
        "license_name": LICENSE_NAME,
        "generated_at": RETRIEVED_AT,
        "facts_sha256": sha256(facts_path),
        "coverage_contract": {
            "crops_minimum": 30,
            "fish_minimum": 50,
            "villagers_minimum": 20,
            "recipes_minimum": 100,
            "standard_bundles_complete": True,
            "acquisition_relations_minimum": 150,
        },
        "notes": [
            "Curated versioned release snapshot for deterministic demos and course evaluation.",
            "Standard Bundles are complete; Remixed Bundle coverage is intentionally represented as partial rather than inferred.",
            "Facts were structured from official Wiki pages and should be re-audited when the target game version changes.",
        ],
    }
    write_json(catalog_root / "snapshot_manifest.json", snapshot_manifest)

    guide_rows = guide_seed_rows()
    write_jsonl(guide_seed_path, guide_rows)

    validation, evaluation = evaluation_records()
    validation_path = evaluation_root / "stardew_validation_v1.jsonl"
    eval_path = evaluation_root / "stardew_eval_v1.jsonl"
    write_jsonl(validation_path, validation)
    write_jsonl(eval_path, evaluation)
    all_eval = [*validation, *evaluation]
    evaluation_manifest = {
        "schema_version": 1,
        "game": "stardew_valley",
        "game_version": GAME_VERSION,
        "validation_count": len(validation),
        "eval_count": len(evaluation),
        "total_count": len(all_eval),
        "language_distribution": dict(Counter(row["language"] for row in all_eval)),
        "category_distribution": dict(Counter(row["category"] for row in all_eval)),
        "intent_distribution": dict(Counter(row["intent"] for row in all_eval)),
        "status_distribution": dict(Counter(row["expected_status"] for row in all_eval)),
        "review_status_distribution": dict(Counter(row["review_status"] for row in all_eval)),
        "human_review_required": True,
        "validation_sha256": sha256(validation_path),
        "eval_sha256": sha256(eval_path),
        "generated_at": RETRIEVED_AT,
        "notes": [
            "All 100 records are machine-validated candidates.",
            "A second human reviewer must verify source support before changing review_status to approved.",
        ],
    }
    write_json(evaluation_root / "manifest_v1.json", evaluation_manifest)

    train_rows, training_validation_rows = grounded_training_records()
    train_path = training_root / "stardew_grounded_train_v1.jsonl"
    training_validation_path = training_root / "stardew_grounded_validation_v1.jsonl"
    write_jsonl(train_path, train_rows)
    write_jsonl(training_validation_path, training_validation_rows)
    training_manifest = {
        "schema_version": 1,
        "game": "stardew_valley",
        "dataset_version": "stardew_grounded_v1",
        "train_count": len(train_rows),
        "validation_count": len(training_validation_rows),
        "train_sha256": sha256(train_path),
        "validation_sha256": sha256(training_validation_path),
        "generation_method": "deterministic_template_from_structured_catalog",
        "formal_evaluation_files_excluded": True,
        "generated_at": RETRIEVED_AT,
    }
    write_json(training_root / "manifest_v1.json", training_manifest)

    report = {
        "status": "generated",
        "catalog": snapshot_manifest,
        "guide_seed_pages": len(guide_rows),
        "evaluation": evaluation_manifest,
        "training": training_manifest,
    }
    write_json(reports_root / "release_data_report.json", report)
    return report


def main() -> None:
    report = write_release_data()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
