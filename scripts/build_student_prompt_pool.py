#!/usr/bin/env python3
"""Build decontaminated train/validation/held-out prompt pools.

Training and validation prompts are sourced only from non-evaluation records.
Assistant messages are removed so a frozen Qwen3-0.6B teacher generates fresh
continuations.  Formal evaluation rows are retained only as ``held_out`` prompts
and can never be selected by the teacher-data training stages.

An optional deterministic Stardew Chinese-prompt augmentation uses tracked
Chinese aliases from the structured catalog.  It creates prompts, not answers,
and excludes every entity appearing in the formal evaluation files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.sft_dataset import validate_messages
from src.data.project_pretraining_corpus import stable_split
from src.utils.io import read_jsonl, write_json, write_jsonl

_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def normalize_question(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^a-z0-9\u3400-\u4dbf\u4e00-\u9fff]+", "", text)


def contains_cjk(value: str) -> bool:
    return bool(_CJK_PATTERN.search(str(value)))


def flatten_entities(value: Any) -> list[str]:
    """Return scalar entity labels from list/dict/string benchmark schemas."""

    if value is None:
        return []
    if isinstance(value, Mapping):
        items: list[str] = []
        for child in value.values():
            items.extend(flatten_entities(child))
        return items
    if isinstance(value, (list, tuple, set)):
        items = []
        for child in value:
            items.extend(flatten_entities(child))
        return items
    rendered = str(value).strip()
    return [rendered] if rendered else []


def prompt_messages(messages: list[dict[str, str]], record_id: str) -> list[dict[str, str]]:
    validated = validate_messages(messages, record_id)
    user_indices = [
        index for index, message in enumerate(validated) if message["role"] == "user"
    ]
    if not user_indices:
        raise ValueError(f"{record_id}: prompt has no user message.")
    result = validated[: max(user_indices) + 1]
    if result[-1]["role"] != "user":
        raise ValueError(f"{record_id}: prompt must end with a user message.")
    return result


def infer_domain(path: Path, record: Mapping[str, Any]) -> str:
    explicit = str(record.get("game") or record.get("domain") or "").strip()
    if explicit:
        return explicit
    return "stardew_valley" if "stardew" in str(path).casefold() else "terraria"


def evaluation_messages(record: Mapping[str, Any], *, domain: str) -> list[dict[str, str]]:
    question = str(record.get("question") or "").strip()
    if not question:
        raise ValueError(f"{record.get('id')}: held-out question is empty.")
    system_prompt = str(record.get("system_prompt") or "").strip()
    if not system_prompt:
        language = str(record.get("language") or "en").casefold()
        system_prompt = (
            "你是一个游戏攻略助手。请准确、简洁地回答用户的问题。"
            if language == "zh"
            else "You are a game-guide assistant. Answer the user accurately and concisely."
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def held_out_records(
    paths: Iterable[Path],
) -> tuple[list[dict[str, Any]], set[str], set[str]]:
    records: list[dict[str, Any]] = []
    questions: set[str] = set()
    entities: set[str] = set()

    for path in paths:
        for index, record in enumerate(read_jsonl(path), start=1):
            question = str(record.get("question") or "").strip()
            if not question:
                continue
            normalized_question = normalize_question(question)
            questions.add(normalized_question)
            for entity in flatten_entities(record.get("entities")):
                normalized = normalize_question(entity)
                if normalized:
                    entities.add(normalized)

            record_id = str(record.get("id") or f"{path.name}:{index}")
            domain = infer_domain(path, record)
            language = str(
                record.get("language") or ("zh" if contains_cjk(question) else "en")
            )
            source_id = f"held_out:{path.name}:{record_id}"
            records.append(
                {
                    "id": f"prompt_{hashlib.sha256(source_id.encode('utf-8')).hexdigest()[:20]}",
                    "split": "held_out",
                    "language": language,
                    "domain": domain,
                    "category": str(record.get("category") or record.get("intent") or "other"),
                    "intent": record.get("intent"),
                    "entities": flatten_entities(record.get("entities")),
                    "source_record_id": record_id,
                    "source_path": str(path),
                    "prompt_type": "formal_held_out_evaluation",
                    "reference_answer": record.get("reference_answer"),
                    "expected_status": record.get("expected_status"),
                    "messages": evaluation_messages(record, domain=domain),
                }
            )

    return records, questions, entities


_CHINESE_TEMPLATES: dict[str, tuple[str, str]] = {
    "crop": ("crop_planning", "{alias}适合在哪些季节种植，成熟需要多久？"),
    "fish": ("fish_availability", "在什么季节、天气、时间和地点可以钓到{alias}？"),
    "villager": ("villager_gifts", "{alias}喜欢哪些礼物？生日是什么时候？"),
    "recipe": ("recipe_ingredients", "{alias}需要哪些材料，如何解锁？"),
    "bundle": ("bundle_community_center", "{alias}需要提交哪些物品？"),
}


def chinese_catalog_prompts(
    catalog_dir: Path,
    *,
    eval_entities: set[str],
    seed: int,
    validation_fraction: float,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    generated: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []

    for path in sorted(catalog_dir.glob("*.jsonl")):
        for index, record in enumerate(read_jsonl(path), start=1):
            record_type = str(record.get("record_type") or "").casefold()
            template_entry = _CHINESE_TEMPLATES.get(record_type)
            if template_entry is None:
                continue
            canonical = str(record.get("name") or "").strip()
            aliases = [
                str(alias).strip()
                for alias in record.get("aliases") or []
                if str(alias).strip() and contains_cjk(str(alias))
            ]
            if not canonical or not aliases:
                continue
            normalized_names = {
                normalize_question(canonical),
                *(normalize_question(alias) for alias in aliases),
            }
            if any(name and name in eval_entities for name in normalized_names):
                rejected.append(
                    {
                        "id": str(record.get("source_catalog_id") or f"{path.name}:{index}"),
                        "reason": "held_out_entity",
                    }
                )
                continue

            category, template = template_entry
            alias = aliases[0]
            question = template.format(alias=alias)
            source_catalog_id = str(
                record.get("source_catalog_id") or f"{path.name}:{index}"
            )
            source_id = f"synthetic_zh:{source_catalog_id}:{question}"
            split = stable_split(
                source_id,
                validation_fraction=validation_fraction,
                seed=seed,
            )
            generated.append(
                {
                    "id": f"prompt_{hashlib.sha256(source_id.encode('utf-8')).hexdigest()[:20]}",
                    "split": split,
                    "language": "zh",
                    "domain": "stardew_valley",
                    "category": category,
                    "intent": category,
                    "entities": [canonical, alias],
                    "source_record_id": source_catalog_id,
                    "source_path": str(path),
                    "prompt_type": "deterministic_bilingual_augmentation",
                    "generation_method": "project_authored_template_from_tracked_alias",
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一个游戏攻略助手。请用中文准确、简洁地回答问题。",
                        },
                        {"role": "user", "content": question},
                    ],
                }
            )

    generated.sort(key=lambda row: (row["split"], row["category"], row["id"]))
    if limit is not None:
        generated = generated[: max(0, limit)]
    return generated, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-input", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-input", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-per-category", type=int)
    parser.add_argument("--augment-stardew-zh", action="store_true")
    parser.add_argument(
        "--stardew-catalog-dir",
        type=Path,
        default=PROJECT_ROOT / "data/stardew/catalog/cleaned",
    )
    parser.add_argument("--augmentation-limit", type=int)
    parser.add_argument("--augmentation-validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    held_out, eval_questions, eval_entities = held_out_records(args.eval_input)
    output: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    category_counts: Counter[tuple[str, str]] = Counter()

    for split, paths in (("train", args.train_input), ("validation", args.validation_input)):
        for path in paths:
            lowered_path = str(path).casefold().replace("\\", "/")
            if "/evaluation/" in lowered_path or "_eval" in Path(path).stem.casefold():
                raise ValueError(f"Formal evaluation path is not allowed: {path}")
            for index, record in enumerate(read_jsonl(path), start=1):
                source_split = str(record.get("split") or split).casefold()
                if source_split in {"eval", "test", "held_out"}:
                    raise ValueError(
                        f"Evaluation record cannot enter prompt pool: {record.get('id')}"
                    )
                record_id = str(record.get("id") or f"{path.name}:{index}")
                messages = record.get("messages")
                if not isinstance(messages, list):
                    rejected.append({"id": record_id, "reason": "missing_messages"})
                    continue
                prompt = prompt_messages(messages, record_id)
                user_text = next(
                    message["content"]
                    for message in reversed(prompt)
                    if message["role"] == "user"
                )
                if normalize_question(user_text) in eval_questions:
                    rejected.append({"id": record_id, "reason": "exact_eval_question"})
                    continue

                domain = infer_domain(path, record)
                language = str(
                    record.get("language") or ("zh" if contains_cjk(user_text) else "en")
                )
                category = str(record.get("category") or record.get("intent") or "other")
                key = (split, category)
                if (
                    args.max_per_category is not None
                    and category_counts[key] >= args.max_per_category
                ):
                    rejected.append({"id": record_id, "reason": "category_limit"})
                    continue
                category_counts[key] += 1

                source_id = f"{path.name}:{record_id}:{split}"
                output.append(
                    {
                        "id": f"prompt_{hashlib.sha256(source_id.encode('utf-8')).hexdigest()[:20]}",
                        "split": split,
                        "language": language,
                        "domain": domain,
                        "category": category,
                        "intent": record.get("intent"),
                        "entities": flatten_entities(record.get("entities")),
                        "source_record_id": record_id,
                        "source_path": str(path),
                        "prompt_type": "instruction_distillation",
                        "messages": prompt,
                    }
                )

    augmentation_count = 0
    if args.augment_stardew_zh:
        augmentation, augmentation_rejections = chinese_catalog_prompts(
            args.stardew_catalog_dir,
            eval_entities=eval_entities,
            seed=args.seed,
            validation_fraction=args.augmentation_validation_fraction,
            limit=args.augmentation_limit,
        )
        augmentation_count = len(augmentation)
        output.extend(augmentation)
        rejected.extend(augmentation_rejections)

    output.extend(held_out)
    output.sort(
        key=lambda row: (
            row["split"],
            row["domain"],
            row["category"],
            row["id"],
        )
    )
    if not output:
        raise RuntimeError("Prompt pool is empty.")
    write_jsonl(args.output, output)

    manifest = {
        "schema_version": 2,
        "output": str(args.output.resolve()),
        "records": len(output),
        "splits": dict(sorted(Counter(row["split"] for row in output).items())),
        "languages": dict(sorted(Counter(row["language"] for row in output).items())),
        "domains": dict(sorted(Counter(row["domain"] for row in output).items())),
        "prompt_types": dict(
            sorted(Counter(row["prompt_type"] for row in output).items())
        ),
        "categories": dict(sorted(Counter(row["category"] for row in output).items())),
        "formal_held_out_records": len(held_out),
        "exact_eval_questions_excluded_from_training": len(eval_questions),
        "held_out_entities": len(eval_entities),
        "deterministic_chinese_augmentation_records": augmentation_count,
        "rejected": {
            "count": len(rejected),
            "reasons": dict(sorted(Counter(row["reason"] for row in rejected).items())),
        },
        "leakage_contract": (
            "Formal evaluation prompts use split=held_out. Teacher-data generation selects "
            "only train/validation. Exact evaluation questions and catalog aliases matching "
            "held-out entities are excluded from training augmentation."
        ),
    }
    write_json(args.manifest, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
