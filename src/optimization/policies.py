"""Scheduling policies for the phase-separated serving simulator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence, TypeVar


T = TypeVar("T", bound="Schedulable")


class Schedulable(Protocol):
    request_id: str
    arrival_time_ms: float
    prompt_tokens: int
    output_tokens: int


class SchedulingPolicy(Protocol[T]):
    name: str

    def select(self, ready: Sequence[T]) -> int:
        """Return the index of the next request in a non-empty ready queue."""


@dataclass(frozen=True)
class FCFSPolicy:
    """First-come-first-served with a stable request-ID tie break."""

    name: str = "fcfs"

    def select(self, ready: Sequence[T]) -> int:
        if not ready:
            raise ValueError("Cannot schedule from an empty queue.")
        return min(
            range(len(ready)),
            key=lambda index: (
                ready[index].arrival_time_ms,
                ready[index].request_id,
            ),
        )


@dataclass(frozen=True)
class ShortestOutputFirstPolicy:
    """Prefer requests with fewer output tokens, then FCFS."""

    name: str = "shortest_output_first"

    def select(self, ready: Sequence[T]) -> int:
        if not ready:
            raise ValueError("Cannot schedule from an empty queue.")
        return min(
            range(len(ready)),
            key=lambda index: (
                ready[index].output_tokens,
                ready[index].arrival_time_ms,
                ready[index].request_id,
            ),
        )


def build_policy(name: str) -> SchedulingPolicy:
    normalized = str(name).strip().casefold().replace("-", "_")
    if normalized in {"fcfs", "fifo"}:
        return FCFSPolicy()
    if normalized in {"shortest_output_first", "sof", "shortest"}:
        return ShortestOutputFirstPolicy()
    raise ValueError(
        "Unsupported scheduling policy: "
        f"{name!r}. Expected fcfs or shortest_output_first."
    )
