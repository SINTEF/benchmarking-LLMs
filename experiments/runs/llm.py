"""Async OpenRouter chat client with retries, 429 handling and usage/cost accounting."""

from __future__ import annotations

import asyncio
import os
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx

from experiments.runs.limiter import RateLimiter

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
FATAL_STATUS = {401, 402, 403}  # bad key, out of credits, forbidden: no point continuing the run


class FatalAPIError(RuntimeError):
    """The whole run must stop (authentication, credits...)."""


class CallError(RuntimeError):
    """This call failed for good (after retries, or a non-retryable 4xx)."""


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int
    cost_usd: float | None
    seconds: float  # successful attempt only
    attempts: int
    provider: str | None


@dataclass
class CallStats:
    """Shared counters the executor reports as progress."""

    calls: int = 0  # HTTP attempts
    retries: int = 0
    rate_limited: int = 0  # 429 responses


def _retry_after(resp: httpx.Response | None) -> float | None:
    if resp is None:
        return None
    value = resp.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def usage_cost(usage: dict[str, Any]) -> float | None:
    """USD actually spent on one call.

    With a BYOK key OpenRouter reports ``cost`` = its own fee only (often 0); the provider bills the
    inference itself, reported as ``cost_details.upstream_inference_cost``.
    """
    if "cost" not in usage:
        return None
    cost = float(usage.get("cost") or 0.0)
    if usage.get("is_byok"):
        cost += float((usage.get("cost_details") or {}).get("upstream_inference_cost") or 0.0)
    return cost


def _backoff(attempt: int) -> float:
    return min(60.0, 2.0 ** attempt) * (0.5 + random.random() / 2)


class OpenRouterClient:
    def __init__(
        self,
        model: str,
        limiter: RateLimiter,
        provider: str | None = None,
        api_key: str | None = None,
        max_retries: int = 6,
        timeout_s: float = 300.0,
        stats: CallStats | None = None,
        url: str = CHAT_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.limiter = limiter
        self.max_retries = max_retries
        self.stats = stats or CallStats()
        self.url = url
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise FatalAPIError("OPENROUTER_API_KEY is not set.")
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}", "X-Title": "benchmarking-LLMs"},
            timeout=httpx.Timeout(timeout_s, connect=20.0),
            limits=httpx.Limits(max_connections=limiter.max_concurrency + 4),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _body(self, prompt: str) -> dict[str, Any]:
        # One user message and no sampling parameters: the same request LionLinker makes.
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "usage": {"include": True},
        }
        if self.provider:
            body["provider"] = {"order": [self.provider], "allow_fallbacks": False}
        return body

    async def chat(self, prompt: str) -> ChatResult:
        last_error = "unknown error"
        for attempt in range(self.max_retries + 1):
            if attempt:
                self.stats.retries += 1
            resp: httpx.Response | None = None
            async with self.limiter:
                started = time.monotonic()
                self.stats.calls += 1
                try:
                    resp = await self._client.post(self.url, json=self._body(prompt))
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    resp = None
                seconds = time.monotonic() - started

            if resp is None:
                await asyncio.sleep(_backoff(attempt))
                continue

            status = resp.status_code
            if status == 429:
                self.stats.rate_limited += 1
                wait = _retry_after(resp) or _backoff(attempt)
                self.limiter.cooldown(wait)
                last_error = f"429 rate limited: {resp.text[:200]}"
                continue
            if status in FATAL_STATUS:
                raise FatalAPIError(f"HTTP {status}: {resp.text[:300]}")
            if status in RETRY_STATUS:
                last_error = f"HTTP {status}: {resp.text[:200]}"
                await asyncio.sleep(_backoff(attempt))
                continue
            if status >= 400:
                raise CallError(f"HTTP {status}: {resp.text[:300]}")

            try:
                data = resp.json()
            except ValueError:
                last_error = f"invalid JSON: {resp.text[:200]}"
                await asyncio.sleep(_backoff(attempt))
                continue
            if data.get("error"):
                # OpenRouter reports some upstream failures with HTTP 200 and an error object.
                err = data["error"]
                code = err.get("code") if isinstance(err, dict) else None
                last_error = f"provider error {code}: {str(err)[:200]}"
                if code == 429:
                    self.stats.rate_limited += 1
                    self.limiter.cooldown(_backoff(attempt))
                    continue
                if isinstance(code, int) and code in FATAL_STATUS:
                    raise FatalAPIError(last_error)
                if isinstance(code, int) and 400 <= code < 500 and code not in RETRY_STATUS:
                    raise CallError(last_error)
                await asyncio.sleep(_backoff(attempt))
                continue
            choices = data.get("choices") or []
            if not choices:
                last_error = "response without choices"
                await asyncio.sleep(_backoff(attempt))
                continue

            usage = data.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            return ChatResult(
                text=(choices[0].get("message") or {}).get("content") or "",
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                reasoning_tokens=int(details.get("reasoning_tokens") or 0),
                cost_usd=usage_cost(usage),
                seconds=seconds,
                attempts=attempt + 1,
                provider=data.get("provider"),
            )
        raise CallError(f"gave up after {self.max_retries + 1} attempts: {last_error}")
