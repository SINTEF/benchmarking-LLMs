"""Async chat client with retries, 429 handling and usage/cost accounting.

Talks to OpenRouter by default, or to any OpenAI-compatible server (vLLM, Ollama, TGI, LM Studio...)
given its base URL: ``<base>/chat/completions``, plain OpenAI request body, optional bearer key.
Keys are passed in by the caller (see ``experiments.runs.endpoints``).
"""

from __future__ import annotations

import asyncio
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
    is_byok: bool = False  # served on your own provider key


@dataclass
class CallStats:
    """Shared counters the executor reports as progress."""

    calls: int = 0  # HTTP attempts
    retries: int = 0
    rate_limited: int = 0  # 429 responses
    shared_rejected: int = 0  # answers served on shared capacity, discarded under "BYOK only"
    shared_rejected_usd: float = 0.0  # what those discarded answers cost


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


def normalize_endpoint(endpoint: str | None) -> str | None:
    """Base URL of an OpenAI-compatible API ("http://gpu01:8000/v1"), or None for OpenRouter."""
    value = (endpoint or "").strip().rstrip("/")
    if not value:
        return None
    if not value.startswith(("http://", "https://")):
        value = f"http://{value}"
    return value.removesuffix("/chat/completions")


def chat_url(endpoint: str) -> str:
    return f"{normalize_endpoint(endpoint)}/chat/completions"


def endpoint_host(endpoint: str) -> str:
    return httpx.URL(endpoint).netloc.decode()


def list_endpoint_models(endpoint: str, api_key: str | None = None, timeout: float = 10.0) -> list[str]:
    """Model ids an OpenAI-compatible server serves (``GET <base>/models``)."""
    resp = httpx.get(f"{normalize_endpoint(endpoint)}/models", timeout=timeout,
                     headers={"Authorization": f"Bearer {api_key}"} if api_key else {})
    resp.raise_for_status()
    return sorted(m["id"] for m in resp.json().get("data", []) if m.get("id"))


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
        provider_prefs: dict[str, Any] | None = None,
        require_byok: bool = False,
        endpoint: str | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        # OpenRouter has no request option forbidding shared capacity: with require_byok an answer
        # not served on your own key is discarded and retried after a back-off.
        self.require_byok = require_byok
        # OpenRouter "provider" object; a pinned ``provider`` tag is shorthand for order=[tag], no fallbacks
        self.provider_prefs = provider_prefs or (
            {"order": [provider], "allow_fallbacks": False} if provider else None
        )
        self.limiter = limiter
        self.max_retries = max_retries
        self.stats = stats or CallStats()
        # A custom OpenAI-compatible endpoint gets a plain OpenAI body: no OpenRouter routing or usage options.
        self.openrouter = not endpoint
        self.url = chat_url(endpoint) if endpoint else url
        if self.openrouter and not api_key:
            raise FatalAPIError("No OpenRouter API key: save one on the Endpoints page (or set OPENROUTER_API_KEY).")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        if self.openrouter:
            headers["X-Title"] = "benchmarking-LLMs"
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=20.0),
            limits=httpx.Limits(max_connections=limiter.max_concurrency + 4),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _body(self, prompt: str) -> dict[str, Any]:
        # One user message and no sampling parameters: the same request LionLinker makes.
        body: dict[str, Any] = {"model": self.model, "messages": [{"role": "user", "content": prompt}]}
        if not self.openrouter:
            return body
        body["usage"] = {"include": True}
        if self.provider_prefs:
            body["provider"] = self.provider_prefs
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
            if self.require_byok and not usage.get("is_byok"):
                self.stats.shared_rejected += 1
                self.stats.shared_rejected_usd += usage_cost(usage) or 0.0
                last_error = f"served by {data.get('provider')} on OpenRouter shared capacity, not your BYOK key"
                # Shared capacity is used when the key is throttled: pause everyone, then retry.
                self.limiter.cooldown(_backoff(attempt))
                continue
            details = usage.get("completion_tokens_details") or {}
            message = choices[0].get("message") or {}
            return ChatResult(
                text=message.get("content") or "",
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                reasoning_tokens=int(details.get("reasoning_tokens") or 0),
                cost_usd=usage_cost(usage),
                seconds=seconds,
                attempts=attempt + 1,
                provider=data.get("provider") if self.openrouter else endpoint_host(self.url),
                is_byok=bool(usage.get("is_byok")),
            )
        raise CallError(f"gave up after {self.max_retries + 1} attempts: {last_error}")
