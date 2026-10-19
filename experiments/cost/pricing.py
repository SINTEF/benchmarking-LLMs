"""Live model prices from OpenRouter's public /models endpoint, cached on disk."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import requests

from experiments.cost.tasks import REPO_ROOT

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
CACHE_PATH = REPO_ROOT / ".cache" / "openrouter_models.json"
MAX_AGE_SECONDS = 24 * 3600


@dataclass(frozen=True)
class ModelPrice:
    id: str
    name: str
    input_per_token: float  # USD
    output_per_token: float  # USD
    per_request: float  # USD
    context_length: int | None
    reasoning: bool  # model emits (billed) reasoning tokens
    provider: str | None = None  # OpenRouter endpoint tag, e.g. "groq" or "deepinfra/bf16"

    @property
    def input_per_mtok(self) -> float:
        return self.input_per_token * 1e6

    @property
    def output_per_mtok(self) -> float:
        return self.output_per_token * 1e6


@dataclass
class PriceBook:
    models: dict[str, ModelPrice]
    source: str  # "live", "cache", "unavailable"
    fetched_at: float | None
    error: str | None = None

    def find(self, model_id: str) -> ModelPrice | None:
        """Exact id, else a unique OpenRouter id ending in '/<model_id>' (e.g. gpt-oss-120b)."""
        if model_id in self.models:
            return self.models[model_id]
        matches = [m for mid, m in self.models.items() if mid.split("/", 1)[-1] == model_id]
        return matches[0] if len(matches) == 1 else None


def _to_float(value) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return f if f >= 0 else 0.0  # OpenRouter uses -1 for router/variable pricing


def parse_models(payload: dict) -> dict[str, ModelPrice]:
    out: dict[str, ModelPrice] = {}
    for item in payload.get("data", []):
        pricing = item.get("pricing") or {}
        params = set(item.get("supported_parameters") or [])
        out[item["id"]] = ModelPrice(
            id=item["id"],
            name=item.get("name") or item["id"],
            input_per_token=_to_float(pricing.get("prompt")),
            output_per_token=_to_float(pricing.get("completion")),
            per_request=_to_float(pricing.get("request")),
            context_length=item.get("context_length"),
            reasoning=bool({"reasoning", "include_reasoning"} & params),
        )
    return out


def load_prices(refresh: bool = False, cache_path: Path = CACHE_PATH, timeout: int = 15) -> PriceBook:
    cached = None
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = None
    if cached and not refresh and time.time() - cached.get("fetched_at", 0) < MAX_AGE_SECONDS:
        return PriceBook(parse_models(cached["payload"]), "cache", cached["fetched_at"])
    try:
        resp = requests.get(OPENROUTER_MODELS_URL, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        now = time.time()
        cache_path.write_text(json.dumps({"fetched_at": now, "payload": payload}), encoding="utf-8")
        return PriceBook(parse_models(payload), "live", now)
    except Exception as exc:  # offline, proxy, rate limit...
        if cached:
            return PriceBook(parse_models(cached["payload"]), "cache", cached["fetched_at"], str(exc))
        return PriceBook({}, "unavailable", None, str(exc))


# --------------------------------------------------------------------------- providers
#
# One OpenRouter model is usually served by many providers at different prices (and quantizations).
# GET /api/v1/models/{id}/endpoints lists them.

ENDPOINTS_URL = "https://openrouter.ai/api/v1/models/{model_id}/endpoints"
ENDPOINTS_CACHE_DIR = REPO_ROOT / ".cache" / "openrouter_endpoints"


@dataclass(frozen=True)
class ProviderPrice:
    provider: str  # display name, e.g. "Groq"
    tag: str  # unique endpoint id, e.g. "groq" or "deepinfra/bf16" (use this to pin a provider)
    quantization: str
    input_per_token: float
    output_per_token: float
    per_request: float
    context_length: int | None
    max_completion_tokens: int | None
    status: int  # 0 = healthy; negative = degraded/down
    uptime_1d: float | None
    latency_s: float | None = None  # median time to first token, last 30 min (when OpenRouter reports it)
    throughput_tps: float | None = None  # median output tokens/s, last 30 min

    @property
    def healthy(self) -> bool:
        return self.status == 0

    def call_cost(self, input_tokens: float, output_tokens: float) -> float:
        return input_tokens * self.input_per_token + output_tokens * self.output_per_token + self.per_request


@dataclass
class ProviderList:
    model_id: str
    providers: list[ProviderPrice]
    source: str  # "live", "cache", "unavailable"
    error: str | None = None

    def cheapest(self, input_tokens: float, output_tokens: float, healthy_only: bool = True) -> ProviderPrice | None:
        pool = [p for p in self.providers if p.healthy] if healthy_only else list(self.providers)
        pool = pool or list(self.providers)
        return min(pool, key=lambda p: p.call_cost(input_tokens, output_tokens)) if pool else None

    def by_tag(self, tag: str) -> ProviderPrice | None:
        for p in self.providers:
            if tag in (p.tag, p.provider):
                return p
        return None


def _p50(value) -> float | None:
    """OpenRouter reports recent stats either as a number or as percentiles ({"p50": ...})."""
    if isinstance(value, dict):
        value = value.get("p50")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def parse_endpoints(payload: dict) -> list[ProviderPrice]:
    out: list[ProviderPrice] = []
    seen: set[str] = set()
    for ep in (payload.get("data") or {}).get("endpoints", []):
        pricing = ep.get("pricing") or {}
        tag = ep.get("tag") or ep.get("provider_name") or "unknown"
        if tag in seen:
            continue
        seen.add(tag)
        out.append(
            ProviderPrice(
                provider=ep.get("provider_name") or tag,
                tag=tag,
                quantization=ep.get("quantization") or "unknown",
                input_per_token=_to_float(pricing.get("prompt")),
                output_per_token=_to_float(pricing.get("completion")),
                per_request=_to_float(pricing.get("request")),
                context_length=ep.get("context_length"),
                max_completion_tokens=ep.get("max_completion_tokens"),
                status=int(ep.get("status") or 0),
                uptime_1d=ep.get("uptime_last_1d"),
                latency_s=(lambda ms: ms / 1000 if ms is not None else None)(_p50(ep.get("latency_last_30m"))),
                throughput_tps=_p50(ep.get("throughput_last_30m")),
            )
        )
    return out


def load_endpoints(
    model_id: str, refresh: bool = False, cache_dir: Path = ENDPOINTS_CACHE_DIR, timeout: int = 15
) -> ProviderList:
    path = cache_dir / (model_id.replace("/", "__") + ".json")
    cached = None
    if path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = None
    if cached and not refresh and time.time() - cached.get("fetched_at", 0) < MAX_AGE_SECONDS:
        return ProviderList(model_id, parse_endpoints(cached["payload"]), "cache")
    try:
        resp = requests.get(ENDPOINTS_URL.format(model_id=model_id), timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_at": time.time(), "payload": payload}), encoding="utf-8")
        return ProviderList(model_id, parse_endpoints(payload), "live")
    except Exception as exc:
        if cached:
            return ProviderList(model_id, parse_endpoints(cached["payload"]), "cache", str(exc))
        return ProviderList(model_id, [], "unavailable", str(exc))


def with_provider(model: ModelPrice, provider: ProviderPrice) -> ModelPrice:
    """The model priced at one specific provider."""
    return ModelPrice(
        id=model.id,
        name=model.name,
        input_per_token=provider.input_per_token,
        output_per_token=provider.output_per_token,
        per_request=provider.per_request,
        context_length=provider.context_length or model.context_length,
        reasoning=model.reasoning,
        provider=provider.tag,
    )


def manual_price(model_id: str, input_per_mtok: float, output_per_mtok: float, reasoning: bool) -> ModelPrice:
    return ModelPrice(model_id, model_id, input_per_mtok / 1e6, output_per_mtok / 1e6, 0.0, None, reasoning)


def price_to_dict(price: ModelPrice) -> dict:
    return asdict(price)
