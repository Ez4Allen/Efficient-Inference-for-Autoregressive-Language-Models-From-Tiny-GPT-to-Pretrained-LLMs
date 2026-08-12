from __future__ import annotations

from pathlib import Path

from src.data.project_pretraining_corpus import ProjectCorpusBuilder, stable_split
from src.utils.io import read_jsonl


def test_stable_split_is_deterministic() -> None:
    first = stable_split("source", validation_fraction=0.2, seed=42)
    second = stable_split("source", validation_fraction=0.2, seed=42)
    assert first == second


def test_builder_deduplicates_and_excludes_eval_entity(tmp_path: Path) -> None:
    evaluation = tmp_path / "eval.jsonl"
    evaluation.write_text(
        '{"id":"e","question":"Where is Blueberry?","entities":["Blueberry"]}\n',
        encoding="utf-8",
    )
    builder = ProjectCorpusBuilder(validation_fraction=0.0, exclude_eval_entities=True)
    builder.add_evaluation_records([evaluation])
    builder.add_document(
        source_id="blueberry",
        text="Blueberry grows in Summer.",
        domain="stardew",
        source_type="catalog",
        entity_name="Blueberry",
    )
    builder.add_document(
        source_id="catfish",
        text="Catfish requires rain.",
        domain="stardew",
        source_type="catalog",
        entity_name="Catfish",
    )
    builder.add_document(
        source_id="catfish-copy",
        text="Catfish requires rain.",
        domain="stardew",
        source_type="catalog",
        entity_name="Catfish",
    )
    output = tmp_path / "corpus.jsonl"
    manifest = tmp_path / "manifest.json"
    report = builder.write(output, manifest)
    rows = read_jsonl(output)
    assert [row["source_id"] for row in rows] == ["catfish"]
    assert report["rejections"]["reasons"]["held_out_entity"] == 1
    assert report["rejections"]["reasons"]["duplicate_text"] == 1


def test_builder_flattens_entity_mapping_and_excludes_alias(tmp_path: Path) -> None:
    evaluation = tmp_path / "eval_mapping.jsonl"
    evaluation.write_text(
        '{"id":"e","question":"gift?","entities":{"npc":["Abigail"]}}\n',
        encoding="utf-8",
    )
    builder = ProjectCorpusBuilder(validation_fraction=0.0, exclude_eval_entities=True)
    builder.add_evaluation_records([evaluation])
    builder.add_document(
        source_id="abigail_alias",
        text="游戏实体中文别名：阿比盖尔。对应的英文标准名称：Abigail。",
        domain="stardew_valley",
        source_type="bilingual_alias_bridge",
        entity_name="Someone Else",
        entity_aliases=["Abigail"],
    )
    output = tmp_path / "corpus.jsonl"
    manifest = tmp_path / "manifest.json"
    report = builder.write(output, manifest)
    assert read_jsonl(output) == []
    assert report["rejections"]["reasons"]["held_out_entity"] == 1
