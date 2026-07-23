"""Run one exact prefill/decode benchmark case."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.benchmark import (  # noqa: E402
    BenchmarkCase,
    load_benchmark_config,
    resolve_torch_dtype,
    run_benchmark_case,
)
from src.models.loader import load_causal_lm  # noqa: E402
from src.utils.io import write_json  # noqa: E402
from src.utils.paths import RESULTS_ROOT, resolve_project_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional benchmark YAML.")
    parser.add_argument("--model", help="Hugging Face ID or local model directory.")
    parser.add_argument("--prompt-length", type=int)
    parser.add_argument("--output-length", type=int)
    parser.add_argument("--prompt-type", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-context-overflow", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "raw" / "single_benchmark.json",
    )
    return parser.parse_args()


def _first(config: dict, key: str, fallback):
    value = config.get(key, fallback)
    if isinstance(value, list):
        if not value:
            return fallback
        return value[0]
    return value


def main() -> None:
    args = parse_args()
    config: dict = {}
    if args.config is not None:
        config = load_benchmark_config(resolve_project_path(args.config))

    model_name = args.model or config.get("model_name")
    if not model_name:
        raise ValueError("Provide --model or a config containing model_name.")

    prompt_length = (
        args.prompt_length
        if args.prompt_length is not None
        else int(_first(config, "prompt_lengths", 128))
    )
    output_length = (
        args.output_length
        if args.output_length is not None
        else int(_first(config, "output_lengths", 32))
    )
    prompt_type = (
        args.prompt_type
        if args.prompt_type is not None
        else str(_first(config, "prompt_types", "technical"))
    )
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(_first(config, "batch_sizes", 1))
    )
    runs = args.runs if args.runs is not None else int(config.get("runs", 5))
    warmup_runs = (
        args.warmup_runs
        if args.warmup_runs is not None
        else int(config.get("warmup_runs", 1))
    )
    dtype_value = args.dtype if args.dtype is not None else config.get("dtype", "auto")
    trust_remote_code = args.trust_remote_code or bool(
        config.get("trust_remote_code", False)
    )

    bundle = load_causal_lm(
        model_name=model_name,
        device=args.device,
        dtype=resolve_torch_dtype(dtype_value),
        trust_remote_code=trust_remote_code,
        local_files_only=args.local_files_only,
    )
    record = run_benchmark_case(
        bundle,
        BenchmarkCase(
            prompt_length=prompt_length,
            output_length=output_length,
            prompt_type=prompt_type,
            batch_size=batch_size,
        ),
        warmup_runs=warmup_runs,
        measured_runs=runs,
        enforce_context_limit=not args.allow_context_overflow,
    )

    output_path = write_json(resolve_project_path(args.output), record)
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print(f"\nSaved benchmark result to: {output_path}")


if __name__ == "__main__":
    main()
