"""Evidence-conditioned prompts shared across supported games."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Mapping

from .evidence_selection import EvidenceSelectionConfig, EvidenceSelectionReport, prepare_evidence
from .schemas import GameGuideResult

PromptMode = Literal["evidence_only", "scaffolded"]


@dataclass(slots=True)
class PreparedPrompt:
    messages: list[Mapping[str, str]]
    result: GameGuideResult
    evidence_report: EvidenceSelectionReport
    prompt_mode: str


def _language(result: GameGuideResult) -> str:
    return "zh" if any("\u4e00" <= char <= "\u9fff" for char in result.question) else "en"


def _system_prompt(language: str) -> str:
    if language == "zh":
        return (
            "你是 GameGuideLM，一个以外部证据为唯一游戏事实来源的攻略语言模型。"
            "先判断证据是否足够，再组织简洁、实用的答案。"
            "不得使用模型记忆补充证据中没有的物品、配方、数值、日期、路线或机制。"
            "每个包含游戏事实的段落末尾必须引用本提示中真实存在的来源编号，例如 [S1]。"
            "证据不足、问题歧义或缺少玩家状态时必须明确说明，禁止猜测。"
            "不要输出思维过程、隐藏分析或 <think> 标签。"
        )
    return (
        "You are GameGuideLM, an evidence-grounded game-guide language model. "
        "First determine whether the supplied evidence is sufficient, then write a concise "
        "and practical answer. Use the evidence as the only source of game facts. Do not "
        "add items, recipes, numbers, schedules, routes, or mechanics from model memory. "
        "Every paragraph containing game facts must end with source identifiers that "
        "actually exist in the prompt, such as [S1]. If evidence is missing, ambiguous, "
        "or requires player state, say so explicitly. Do not reveal chain-of-thought or "
        "emit <think> tags."
    )


def _user_prompt(result: GameGuideResult, *, prompt_mode: PromptMode) -> str:
    package = result.context_payload or {
        "game": result.game,
        "question": result.question,
        "intent": result.intent,
        "entity": result.entity,
        "status": result.status,
        "facts": result.facts,
        "warnings": result.warnings,
        "evidence": [item.to_dict() for item in result.evidence],
    }
    language = _language(result)
    if language == "zh":
        parts = [
            f"游戏：{result.game}",
            f"问题：{result.question}",
            "",
            "证据包：",
            json.dumps(package, ensure_ascii=False, indent=2),
        ]
        if prompt_mode == "scaffolded":
            parts.extend([
                "",
                "确定性决策脚手架（仅用于组织答案，不是额外来源）：",
                result.answer,
            ])
        parts.extend([
            "",
            "请用中文生成最终答案。只保留对用户有用的事实，并在相应段落末尾标注来源编号。",
        ])
        return "\n".join(parts)

    parts = [
        f"Game: {result.game}",
        f"Question: {result.question}",
        "",
        "Evidence package:",
        json.dumps(package, ensure_ascii=False, indent=2),
    ]
    if prompt_mode == "scaffolded":
        parts.extend([
            "",
            "Deterministic decision scaffold (for organization only; not an additional source):",
            result.answer,
        ])
    parts.extend([
        "",
        "Write the final answer in English. Keep only useful supported facts and cite the relevant source identifiers at the end of each factual paragraph.",
    ])
    return "\n".join(parts)


def prepare_gameguide_prompt(
    result: GameGuideResult,
    *,
    prompt_mode: PromptMode = "evidence_only",
    evidence_config: EvidenceSelectionConfig | None = None,
) -> PreparedPrompt:
    """Prepare a bounded prompt without mutating the retrieval result.

    ``evidence_only`` is the default research condition. It asks the model to
    reason from retrieved facts rather than rewrite the deterministic answer.
    ``scaffolded`` remains available as an explicit ablation.
    """

    if prompt_mode not in {"evidence_only", "scaffolded"}:
        raise ValueError("prompt_mode must be evidence_only or scaffolded.")
    prepared = prepare_evidence(result, evidence_config)
    messages = [
        {"role": "system", "content": _system_prompt(_language(prepared.result))},
        {"role": "user", "content": _user_prompt(prepared.result, prompt_mode=prompt_mode)},
    ]
    return PreparedPrompt(
        messages=messages,
        result=prepared.result,
        evidence_report=prepared.report,
        prompt_mode=prompt_mode,
    )


def build_gameguide_messages(
    result: GameGuideResult,
    *,
    prompt_mode: PromptMode = "evidence_only",
    evidence_config: EvidenceSelectionConfig | None = None,
) -> list[Mapping[str, str]]:
    return prepare_gameguide_prompt(
        result,
        prompt_mode=prompt_mode,
        evidence_config=evidence_config,
    ).messages


def build_repair_messages(
    prepared: PreparedPrompt,
    invalid_answer: str,
    issues: list[str],
) -> list[Mapping[str, str]]:
    valid_sources = [item.source_id for item in prepared.result.evidence]
    language = _language(prepared.result)
    if language == "zh":
        instruction = (
            "上一个答案未通过证据校验。请重写答案，不要增加新事实。"
            f"校验问题：{', '.join(issues)}。"
            f"允许引用的来源只有：{', '.join(valid_sources) or '无'}。"
            "每个事实段落都必须使用合法来源编号；不要解释校验过程。"
        )
    else:
        instruction = (
            "The previous answer failed grounding validation. Rewrite it without adding "
            f"new facts. Validation issues: {', '.join(issues)}. The only allowed source "
            f"identifiers are: {', '.join(valid_sources) or 'none'}. Every factual paragraph "
            "must use a valid identifier. Do not discuss the validation process."
        )
    return [
        *prepared.messages,
        {"role": "assistant", "content": invalid_answer},
        {"role": "user", "content": instruction},
    ]
