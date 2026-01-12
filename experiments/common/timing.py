"""Timing utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass


@dataclass
class Timer:
    start: float | None = None
    end: float | None = None

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        if self.start is None:
            return 0.0
        end = self.end if self.end is not None else time.perf_counter()
        return (end - self.start) * 1000.0


@contextmanager
def timed() -> Timer:
    timer = Timer()
    with timer:
        yield timer
