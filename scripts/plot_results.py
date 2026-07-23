"""Plot benchmark or serving-simulation results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import iter_jsonl  # noqa: E402
from src.utils.paths import RESULTS_ROOT, resolve_project_path  # noqa: E402


def _load(path: Path) -> tuple[str, list[dict[str, Any]]]:
    if path.suffix.casefold() == ".jsonl":
        return "benchmark", list(iter_jsonl(path))

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "benchmark" in payload and "case" in payload:
        return "benchmark", [payload]
    if isinstance(payload, dict) and "summary" in payload and "requests" in payload:
        return "simulation", [payload]
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return "benchmark", payload
    raise ValueError(f"Unrecognized result schema: {path}")


def _benchmark_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        case = record["case"]
        model = record["model"]
        metrics = record["benchmark"]["metrics"]
        rows.append(
            {
                "model": model["name"],
                "device": model["device"],
                "dtype": model["dtype"],
                "prompt_length": case["prompt_length"],
                "output_length": case["output_length"],
                "prompt_type": case["prompt_type"],
                "batch_size": case["batch_size"],
                "ttft_ms": metrics["ttft_seconds"]["mean"] * 1000.0,
                "tpot_ms": metrics["mean_tpot_seconds"]["mean"] * 1000.0,
                "latency_ms": metrics["total_latency_seconds"]["mean"] * 1000.0,
                "throughput_tokens_per_second": metrics[
                    "throughput_tokens_per_second"
                ]["mean"],
                "peak_allocated_mb": metrics["peak_allocated_bytes"]["maximum"]
                / (1024.0 * 1024.0),
            }
        )
    return pd.DataFrame(rows)


def _plot_grouped_lines(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    group_columns: list[str],
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(9, 5.5))
    for group_key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        label = ", ".join(
            f"{column}={value}" for column, value in zip(group_columns, group_key)
        )
        aggregate = group.groupby(x, as_index=False)[y].mean().sort_values(x)
        axis.plot(aggregate[x], aggregate[y], marker="o", label=label)
    axis.set_title(title)
    axis.set_xlabel(x.replace("_", " ").title())
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    if frame[group_columns].drop_duplicates().shape[0] <= 12:
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def plot_benchmark(records: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    frame = _benchmark_frame(records)
    if frame.empty:
        raise ValueError("No benchmark records to plot.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "benchmark_summary.csv"
    frame.to_csv(summary_path, index=False)

    group = ["model", "prompt_type", "batch_size", "output_length"]
    ttft_path = output_dir / "ttft_vs_prompt_length.png"
    _plot_grouped_lines(
        frame,
        x="prompt_length",
        y="ttft_ms",
        group_columns=group,
        title="Time to First Token vs Prompt Length",
        ylabel="TTFT (ms)",
        output=ttft_path,
    )

    throughput_path = output_dir / "throughput_vs_prompt_length.png"
    _plot_grouped_lines(
        frame,
        x="prompt_length",
        y="throughput_tokens_per_second",
        group_columns=group,
        title="Generation Throughput vs Prompt Length",
        ylabel="Tokens / second",
        output=throughput_path,
    )

    tpot_path = output_dir / "tpot_vs_output_length.png"
    _plot_grouped_lines(
        frame,
        x="output_length",
        y="tpot_ms",
        group_columns=["model", "prompt_type", "batch_size", "prompt_length"],
        title="Mean Time per Output Token",
        ylabel="TPOT (ms/token)",
        output=tpot_path,
    )
    return [summary_path, ttft_path, throughput_path, tpot_path]


def plot_simulation(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    frame = pd.DataFrame(payload["requests"])
    if frame.empty:
        raise ValueError("Simulation result contains no requests.")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "simulation_requests.csv"
    frame.to_csv(summary_path, index=False)

    latency_path = output_dir / "simulation_latency_distribution.png"
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(frame["latency_ms"], bins=min(30, max(5, len(frame) // 4)))
    axis.set_title("Request Latency Distribution")
    axis.set_xlabel("Latency (ms)")
    axis.set_ylabel("Requests")
    figure.tight_layout()
    figure.savefig(latency_path, dpi=160)
    plt.close(figure)

    timeline_path = output_dir / "simulation_completion_timeline.png"
    figure, axis = plt.subplots(figsize=(9, 5))
    ordered = frame.sort_values("arrival_time_ms")
    axis.scatter(ordered["arrival_time_ms"], ordered["completion_ms"], s=18)
    axis.set_title("Arrival vs Completion Time")
    axis.set_xlabel("Arrival time (ms)")
    axis.set_ylabel("Completion time (ms)")
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(timeline_path, dpi=160)
    plt.close(figure)
    return [summary_path, latency_path, timeline_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_ROOT / "figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)
    kind, records = _load(input_path)
    if kind == "benchmark":
        outputs = plot_benchmark(records, output_dir)
    else:
        outputs = plot_simulation(records[0], output_dir)
    print(f"Loaded {kind} results from: {input_path}")
    for output in outputs:
        print(f"Created: {output}")


if __name__ == "__main__":
    main()
