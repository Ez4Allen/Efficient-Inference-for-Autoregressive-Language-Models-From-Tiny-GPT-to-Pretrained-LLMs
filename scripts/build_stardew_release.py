#!/usr/bin/env python3
"""Build, validate, evaluate, and package the Stardew Valley course release.

One command regenerates all deterministic data, audits the legacy SFT candidate
pool, rebuilds both SQLite stores, runs the 100-example regression suite, writes
showcase outputs, validates the release contract, and optionally runs tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results" / "stardew"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_step(name: str, command: list[str], *, logs: Path) -> dict[str, Any]:
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - started
    log_path = logs / f"{name}.log"
    log_path.write_text(
        f"$ {' '.join(command)}\n\nSTDOUT\n{result.stdout}\n\nSTDERR\n{result.stderr}\n",
        encoding="utf-8",
    )
    payload = {
        "name": name,
        "command": command,
        "returncode": result.returncode,
        "seconds": round(elapsed, 4),
        "log": str(log_path.relative_to(PROJECT_ROOT)),
    }
    if result.returncode != 0:
        raise RuntimeError(
            f"Stardew release step {name!r} failed. See {log_path}.\n"
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return payload


def build(*, run_tests: bool) -> dict[str, Any]:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    logs = RESULTS_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    steps = [
        ("generate_release_data", [python, "scripts/generate_stardew_release_data.py"]),
        ("clean_legacy_sft", [python, "scripts/clean_stardew_sft_data.py"]),
        ("build_catalog", [python, "scripts/build_stardew_knowledge.py", "--quiet"]),
        ("build_guides", [python, "scripts/build_stardew_guides.py", "--seed", "--quiet"]),
        (
            "evaluate_regression",
            [
                python,
                "scripts/evaluate_gameguidelm.py",
                "--input",
                "data/stardew/evaluation/stardew_validation_v1.jsonl",
                "data/stardew/evaluation/stardew_eval_v1.jsonl",
                "--output",
                "results/stardew/evaluation_details.jsonl",
                "--summary",
                "results/stardew/evaluation_summary.json",
            ],
        ),
        ("validate_release", [python, "scripts/validate_stardew_release.py"]),
        ("build_demo_outputs", [python, "scripts/demo_stardew_showcase.py"]),
        ("build_html_showcase", [python, "scripts/build_stardew_showcase.py"]),
    ]
    if run_tests:
        steps.append(("pytest", [python, "-m", "pytest", "-q"]))

    started = time.perf_counter()
    completed = [run_step(name, command, logs=logs) for name, command in steps]

    key_files = [
        PROJECT_ROOT / "data" / "stardew" / "catalog" / "cleaned" / "facts.jsonl",
        PROJECT_ROOT / "data" / "stardew" / "catalog" / "snapshot_manifest.json",
        PROJECT_ROOT / "data" / "stardew" / "evaluation" / "stardew_validation_v1.jsonl",
        PROJECT_ROOT / "data" / "stardew" / "evaluation" / "stardew_eval_v1.jsonl",
        PROJECT_ROOT / "data" / "stardew" / "guides" / "seed" / "pages.jsonl",
        PROJECT_ROOT / "data" / "stardew" / "training" / "stardew_grounded_train_v1.jsonl",
        PROJECT_ROOT / "results" / "stardew" / "evaluation_summary.json",
        PROJECT_ROOT / "results" / "stardew" / "release_validation.json",
        PROJECT_ROOT / "demo" / "stardew_showcase.html",
    ]
    manifest = {
        "status": "passed",
        "release": "stardew_course_release_v1",
        "generated_by": "scripts/build_stardew_release.py",
        "total_seconds": round(time.perf_counter() - started, 4),
        "tests_run": run_tests,
        "steps": completed,
        "artifacts": {
            str(path.relative_to(PROJECT_ROOT)): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in key_files
        },
        "truth_boundary": {
            "deterministic_regression": "complete",
            "engineering_validation": "passed",
            "formal_human_review": "pending",
            "qwen_gpu_training": "not_executed_in_offline_release_build",
            "speculative_gpu_benchmark": "not_executed_in_offline_release_build",
        },
    }
    output = RESULTS_ROOT / "release_build_manifest.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-tests", action="store_true", help="Skip the full pytest suite.")
    args = parser.parse_args()
    manifest = build(run_tests=not args.skip_tests)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
