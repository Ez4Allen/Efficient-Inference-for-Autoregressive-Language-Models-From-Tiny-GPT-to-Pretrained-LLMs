"""Run a resumable benchmark sweep from a YAML configuration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark import (  # noqa: E402
    iter_benchmark_cases,
    load_benchmark_config,
    resolve_torch_dtype,
    run_benchmark_case,
)
from src.models.loader import load_causal_lm  # noqa: E402
from src.utils.io import append_jsonl, iter_jsonl  # noqa: E402
from src.utils.paths import RESULTS_ROOT, resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--allow-context-overflow", action="store_true")
    return parser.parse_args()


def _case_key(record: dict) -> tuple[int, int, str, int]:
    case = record["case"]
    return (
        int(case["prompt_length"]),
        int(case["output_length"]),
        str(case["prompt_type"]).casefold(),
        int(case["batch_size"]),
    )


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    config = load_benchmark_config(config_path)
    output_path = resolve_project_path(
        args.output
        if args.output is not None
        else RESULTS_ROOT / "raw" / f"{config_path.stem}_benchmark.jsonl"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    completed: set[tuple[int, int, str, int]] = set()
    if args.resume and output_path.exists():
        completed = {_case_key(record) for record in iter_jsonl(output_path)}
    elif output_path.exists():
        output_path.unlink()

    bundle = load_causal_lm(
        model_name=config["model_name"],
        device=args.device,
        dtype=resolve_torch_dtype(config.get("dtype", "auto")),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        local_files_only=args.local_files_only,
    )

    cases = list(iter_benchmark_cases(config))
    if args.limit_cases is not None:
        if args.limit_cases < 1:
            raise ValueError("--limit-cases must be at least 1.")
        cases = cases[: args.limit_cases]

    completed_this_run = 0
    for index, case in enumerate(cases, start=1):
        if case.key in completed:
            print(f"[{index}/{len(cases)}] skip existing case {case.key}")
            continue

        print(f"[{index}/{len(cases)}] run case {case.key}")
        record = run_benchmark_case(
            bundle,
            case,
            warmup_runs=int(config["warmup_runs"]),
            measured_runs=int(config["runs"]),
            enforce_context_limit=not args.allow_context_overflow,
        )
        append_jsonl(output_path, record)
        completed.add(case.key)
        completed_this_run += 1

        metrics = record["benchmark"]["metrics"]
        print(
            "  TTFT="
            f"{metrics['ttft_seconds']['mean'] * 1000:.3f} ms, "
            "TPOT="
            f"{metrics['mean_tpot_seconds']['mean'] * 1000:.3f} ms, "
            "throughput="
            f"{metrics['throughput_tokens_per_second']['mean']:.2f} tok/s"
        )

    print(
        json.dumps(
            {
                "output": str(output_path),
                "configured_cases": len(cases),
                "completed_this_run": completed_this_run,
                "total_records": len(completed),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
