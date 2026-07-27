from pathlib import Path
import re

import pytest

from src.evaluation.model_benchmark import summarize_benchmark_rows, validate_engines
from src.gameguide import __version__


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_release_version_matches_pyproject() -> None:
    text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match is not None
    assert __version__ == match.group(1)


def test_model_benchmark_summary_and_engine_contract() -> None:
    assert validate_engines(["target", "draft", "target"]) == ("target", "draft")
    with pytest.raises(ValueError, match="target engine is required"):
        validate_engines(["speculative"])

    rows = [
        {
            "example_id": "a",
            "engine": "target",
            "warmup": False,
            "total_time_seconds": 4.0,
            "ttft_seconds": 1.0,
            "mean_tpot_seconds": 0.1,
            "tokens_per_second": 10.0,
            "prompt_tokens": 100,
            "generated_tokens": 40,
            "target_forward_calls": 40,
            "draft_forward_calls": 0,
            "acceptance_rate": 0.0,
            "grounding_valid": True,
            "exact_target_match": None,
        },
        {
            "example_id": "a",
            "engine": "speculative",
            "warmup": False,
            "total_time_seconds": 2.0,
            "ttft_seconds": 1.0,
            "mean_tpot_seconds": 0.04,
            "tokens_per_second": 20.0,
            "prompt_tokens": 100,
            "generated_tokens": 40,
            "target_forward_calls": 20,
            "draft_forward_calls": 50,
            "acceptance_rate": 0.6,
            "grounding_valid": True,
            "exact_target_match": True,
        },
        {
            "example_id": "warm",
            "engine": "target",
            "warmup": True,
            "total_time_seconds": 99.0,
        },
    ]
    summary = summarize_benchmark_rows(rows)
    assert summary["measured_rows"] == 2
    assert summary["engines"]["target"]["runs"] == 1
    assert summary["engines"]["speculative"]["exact_target_match_rate"] == 1.0
    assert summary["target_over_speculative_speedup"] == 2.0
