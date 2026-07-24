"""Prompt construction for evidence-grounded Terraria answers."""

from __future__ import annotations

import json
from typing import Mapping

from .schemas import ContextBundle


def build_grounded_messages(
    context: ContextBundle,
    fallback_answer: str,
) -> list[Mapping[str, str]]:
    if context.language == "zh":
        system = (
            "你是一个只根据提供证据回答的 Terraria 助手。"
            "不得使用模型记忆补充证据中没有的物品、NPC、配方、掉落率、Boss 顺序或机制。"
            "每个包含事实的段落必须在末尾引用一个或多个有效来源编号，例如 [S1]。"
            "只能使用证据包里出现的来源编号。不要输出思维过程或 <think> 标签。"
            "若证据不足，直接采用已验证的确定性回答。"
        )
        user = (
            f"问题：\n{context.payload.get('question')}\n\n"
            f"已验证的确定性回答：\n{fallback_answer}\n\n"
            "证据包：\n"
            + json.dumps(context.payload, ensure_ascii=False, indent=2)
            + "\n\n请用中文给出简洁、自然且带来源编号的最终回答。"
        )
    else:
        system = (
            "You are a Terraria assistant that may answer only from the supplied evidence. "
            "Do not use model memory to add items, NPCs, recipes, drop rates, boss order, "
            "or mechanics absent from the evidence. Every factual paragraph must end with "
            "one or more valid source identifiers such as [S1]. Cite only identifiers present "
            "in the evidence package. Do not reveal chain-of-thought or emit <think> tags. "
            "When evidence is insufficient, preserve the verified deterministic answer."
        )
        user = (
            f"Question:\n{context.payload.get('question')}\n\n"
            f"Verified deterministic answer:\n{fallback_answer}\n\n"
            "Evidence package:\n"
            + json.dumps(context.payload, ensure_ascii=False, indent=2)
            + "\n\nWrite a concise, natural final answer with source identifiers."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
