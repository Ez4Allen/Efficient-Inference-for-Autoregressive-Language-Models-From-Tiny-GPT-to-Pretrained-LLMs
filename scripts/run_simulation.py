#!/usr/bin/env python3
"""Run a phase-separated prefill/decode serving simulation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.optimization import (
    ServingSimulator,
    SimulationConfig,
    generate_poisson_workload,
)
from src.utils.paths import RESULTS_ROOT


def load_config(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Simulation config must be a mapping: {path}")
    return payload


def run_simulation(config_path: Path, output_path: Path) -> dict:
    config = load_config(config_path)
    workload_config = config.get("workload", {})
    system_config = config.get("system", {})
    policy_config = config.get("policy", {})

    arrival_pattern = workload_config.get("arrival_pattern", "poisson")
    if arrival_pattern != "poisson":
        raise ValueError("Only the poisson arrival pattern is currently supported.")

    requests = generate_poisson_workload(
        num_requests=int(workload_config.get("num_requests", 100)),
        arrival_rate_per_second=float(
            workload_config.get("arrival_rate_per_second", 4.0)
        ),
        prompt_lengths=[
            int(value)
            for value in workload_config.get(
                "prompt_lengths", [64, 128, 256, 512, 1024]
            )
        ],
        output_lengths=[
            int(value)
            for value in workload_config.get(
                "output_lengths", [32, 64, 128, 256]
            )
        ],
        seed=int(workload_config.get("seed", 42)),
    )

    simulation_config = SimulationConfig(
        num_prefill_workers=int(system_config.get("num_prefill_workers", 1)),
        num_decode_workers=int(system_config.get("num_decode_workers", 1)),
        prefill_tokens_per_second=float(
            system_config.get("prefill_tokens_per_second", 8000.0)
        ),
        decode_tokens_per_second=float(
            system_config.get("decode_tokens_per_second", 80.0)
        ),
        prefill_fixed_overhead_ms=float(
            system_config.get("prefill_fixed_overhead_ms", 0.0)
        ),
        kv_transfer_overhead_ms=float(
            system_config.get("kv_transfer_overhead_ms", 0.0)
        ),
        policy_name=str(policy_config.get("name", "fcfs")),
    )

    result = ServingSimulator(simulation_config).run(requests)
    payload = result.to_dict()
    payload["config_path"] = str(config_path.resolve())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "simulation.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_ROOT / "simulation" / "fcfs.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_simulation(args.config, args.output)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Saved simulation to: {args.output}")


if __name__ == "__main__":
    main()
