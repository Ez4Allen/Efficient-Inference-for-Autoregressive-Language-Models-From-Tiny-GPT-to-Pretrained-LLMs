"""Reusable stage scheduler for deterministic serving simulations."""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

from .policies import SchedulingPolicy


T = TypeVar("T")


@dataclass(frozen=True)
class ScheduledJob(Generic[T]):
    payload: T
    worker_id: int
    start_time_ms: float
    finish_time_ms: float


class Scheduler(Generic[T]):
    """Schedule ready-time jobs on identical workers with a policy."""

    def __init__(self, *, num_workers: int, policy: SchedulingPolicy) -> None:
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1.")
        self.num_workers = num_workers
        self.policy = policy

    def schedule(
        self,
        jobs: Iterable[T],
        *,
        ready_time: Callable[[T], float],
        duration_ms: Callable[[T], float],
    ) -> list[ScheduledJob[T]]:
        pending = sorted(
            list(jobs),
            key=lambda job: (ready_time(job), getattr(job, "request_id", "")),
        )
        if not pending:
            return []

        workers = [(0.0, worker_id) for worker_id in range(self.num_workers)]
        heapq.heapify(workers)
        ready: list[T] = []
        scheduled: list[ScheduledJob[T]] = []
        next_pending = 0

        while next_pending < len(pending) or ready:
            worker_available, worker_id = heapq.heappop(workers)

            while (
                next_pending < len(pending)
                and ready_time(pending[next_pending]) <= worker_available
            ):
                ready.append(pending[next_pending])
                next_pending += 1

            if not ready:
                next_ready = ready_time(pending[next_pending])
                worker_available = max(worker_available, next_ready)
                while (
                    next_pending < len(pending)
                    and ready_time(pending[next_pending]) <= worker_available
                ):
                    ready.append(pending[next_pending])
                    next_pending += 1

            selected_index = self.policy.select(ready)
            job = ready.pop(selected_index)
            duration = float(duration_ms(job))
            if duration < 0:
                raise ValueError("Job duration cannot be negative.")

            start = max(worker_available, float(ready_time(job)))
            finish = start + duration
            scheduled.append(
                ScheduledJob(
                    payload=job,
                    worker_id=worker_id,
                    start_time_ms=start,
                    finish_time_ms=finish,
                )
            )
            heapq.heappush(workers, (finish, worker_id))

        return scheduled
