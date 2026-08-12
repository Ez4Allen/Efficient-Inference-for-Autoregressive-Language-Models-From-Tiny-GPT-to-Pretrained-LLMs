from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.utils.io import read_jsonl


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_student_prompt_pool.py"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_prompt_pool_keeps_formal_eval_held_out(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    evaluation = tmp_path / "formal_eval.jsonl"
    output = tmp_path / "prompts.jsonl"
    manifest = tmp_path / "manifest.json"

    write_jsonl(
        train,
        [
            {
                "id": "train_1",
                "split": "train",
                "messages": [
                    {"role": "user", "content": "How do I obtain Wood?"},
                    {"role": "assistant", "content": "Answer"},
                ],
            }
        ],
    )
    write_jsonl(
        validation,
        [
            {
                "id": "validation_1",
                "split": "validation",
                "messages": [
                    {"role": "user", "content": "How do I obtain Stone?"},
                    {"role": "assistant", "content": "Answer"},
                ],
            }
        ],
    )
    write_jsonl(
        evaluation,
        [
            {
                "id": "eval_1",
                "split": "eval",
                "game": "stardew_valley",
                "language": "zh",
                "question": "阿比盖尔喜欢什么礼物？",
                "reference_answer": "她喜欢紫水晶。",
                "entities": ["Abigail"],
            }
        ],
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train-input",
            str(train),
            "--validation-input",
            str(validation),
            "--eval-input",
            str(evaluation),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
        check=True,
    )

    rows = read_jsonl(output)
    assert {row["split"] for row in rows} == {"train", "validation", "held_out"}
    held_out = next(row for row in rows if row["split"] == "held_out")
    assert held_out["language"] == "zh"
    assert held_out["reference_answer"] == "她喜欢紫水晶。"
    assert held_out["prompt_type"] == "formal_held_out_evaluation"
    assert all(
        row["source_record_id"] != "eval_1"
        for row in rows
        if row["split"] in {"train", "validation"}
    )
