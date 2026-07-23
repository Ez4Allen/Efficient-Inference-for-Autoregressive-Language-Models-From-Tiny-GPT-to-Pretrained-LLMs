"""Portable accelerator and memory monitoring utilities."""

from __future__ import annotations

from typing import Any

import torch

from src.utils.device import memory_stats, reset_peak_memory, resolve_device


def get_gpu_info(
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable accelerator description.

    On CPU/MPS systems the function returns ``available=False`` instead of raising.
    CUDA-only fields are populated when a CUDA device is selected.
    """

    resolved = resolve_device(device)
    result: dict[str, Any] = {
        "available": resolved.type == "cuda",
        "device": str(resolved),
        "device_type": resolved.type,
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "name": None,
        "compute_capability": None,
        "supports_bfloat16": False,
        "memory": memory_stats(resolved),
    }

    if resolved.type == "cuda":
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "supports_bfloat16": bool(torch.cuda.is_bf16_supported()),
            }
        )
    elif resolved.type == "mps":
        result["name"] = "Apple Metal Performance Shaders"

    return result


def begin_memory_measurement(
    device: str | torch.device | None = None,
) -> None:
    """Reset peak-memory counters before a measured section."""

    reset_peak_memory(device)


def end_memory_measurement(
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Return the current and peak memory snapshot after a measured section."""

    return memory_stats(device)
