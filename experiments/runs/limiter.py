"""Client-side rate limiting for LLM requests: concurrency cap, optional RPM cap, 429 cooldown."""

from __future__ import annotations

import asyncio
import time


class RateLimiter:
    """``async with limiter:`` around every request.

    * at most ``max_concurrency`` requests in flight;
    * requests start at least ``60 / max_rpm`` seconds apart (when ``max_rpm`` > 0);
    * after a 429, ``cooldown(s)`` pauses *all* new requests, not just the one that was throttled,
      so the provider sees the whole run back off.
    """

    def __init__(self, max_concurrency: int, max_rpm: int = 0) -> None:
        self.max_concurrency = max_concurrency
        self.max_rpm = max_rpm
        self._sem = asyncio.Semaphore(max_concurrency)
        self._interval = 60.0 / max_rpm if max_rpm > 0 else 0.0
        self._next_start = 0.0
        self._cooldown_until = 0.0
        self._lock = asyncio.Lock()
        self.in_flight = 0

    async def __aenter__(self) -> "RateLimiter":
        await self._sem.acquire()
        try:
            async with self._lock:  # serialise start times so the RPM spacing holds
                while True:
                    now = time.monotonic()
                    wait = max(self._cooldown_until, self._next_start) - now
                    if wait <= 0:
                        break
                    await asyncio.sleep(wait)
                self._next_start = max(now, self._next_start) + self._interval
        except BaseException:
            self._sem.release()
            raise
        self.in_flight += 1
        return self

    async def __aexit__(self, *exc) -> None:
        self.in_flight -= 1
        self._sem.release()

    def cooldown(self, seconds: float) -> None:
        self._cooldown_until = max(self._cooldown_until, time.monotonic() + seconds)

    @property
    def cooling_down(self) -> bool:
        return time.monotonic() < self._cooldown_until
