"""Device selection, synchronization, and memory helpers."""

from __future__ import annotations

from typing import Any

import torch


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve ``auto``/``None`` to CUDA, MPS, or CPU in that order."""

    if isinstance(device, torch.device):
        resolved = device
    else:
        requested = "auto" if device is None else str(device).strip().casefold()
        if requested in {"", "auto"}:
            if torch.cuda.is_available():
                resolved = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                resolved = torch.device("mps")
            else:
                resolved = torch.device("cpu")
        else:
            resolved = torch.device(requested)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is not available.")
    if resolved.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is None or not mps_backend.is_available():
            raise RuntimeError("MPS was requested, but MPS is not available.")
    return resolved


def synchronize(device: str | torch.device | None = None) -> None:
    """Synchronize asynchronous accelerator work when supported."""

    resolved = resolve_device(device)
    if resolved.type == "cuda":
        torch.cuda.synchronize(resolved)
    elif resolved.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def reset_peak_memory(device: str | torch.device | None = None) -> None:
    """Reset CUDA peak-memory statistics; no-op on non-CUDA devices."""

    resolved = resolve_device(device)
    if resolved.type == "cuda":
        torch.cuda.reset_peak_memory_stats(resolved)


def memory_stats(device: str | torch.device | None = None) -> dict[str, Any]:
    """Return a portable memory snapshot for the selected device."""

    resolved = resolve_device(device)
    result: dict[str, Any] = {
        "device": str(resolved),
        "device_type": resolved.type,
        "allocated_bytes": 0,
        "reserved_bytes": 0,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "total_bytes": None,
        "free_bytes": None,
    }
    if resolved.type != "cuda":
        return result

    free_bytes, total_bytes = torch.cuda.mem_get_info(resolved)
    result.update(
        {
            "allocated_bytes": torch.cuda.memory_allocated(resolved),
            "reserved_bytes": torch.cuda.memory_reserved(resolved),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(resolved),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(resolved),
            "total_bytes": total_bytes,
            "free_bytes": free_bytes,
        }
    )
    return result
