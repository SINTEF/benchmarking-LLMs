"""Which OpenRouter providers run on your own keys (BYOK), and the routing that prefers them.

OpenRouter tries prioritized BYOK keys first only when a request names no provider ``order``; when
the BYOK key is rate limited it silently falls back to shared capacity (other providers, other
quantizations, OpenRouter credits). For a benchmark we usually want the opposite: stay on our keys
and back off on 429 instead. Policies (``RunConfig.provider_policy``):

* ``byok``        - only BYOK providers serving the model (``order`` + ``allow_fallbacks: false``).
                    OpenRouter may still use that provider's *shared* endpoint when your key is
                    throttled; only the key setting "Never use shared capacity" prevents that.
* ``byok_first``  - BYOK providers first, then OpenRouter may fall back to shared capacity
* ``routing``     - no provider preference: OpenRouter decides (BYOK first, then shared)
* ``pin``         - one endpoint tag (``RunConfig.provider``), no fallbacks

BYOK providers are found, in order, from ``OPENROUTER_BYOK_PROVIDERS`` (comma-separated slugs,
e.g. ``cerebras,groq``), the BYOK list API (needs ``OPENROUTER_MANAGEMENT_KEY``), and calls seen
in earlier runs (OpenRouter flags them ``is_byok``).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import requests

from experiments.cost.pricing import ProviderList
from experiments.cost.tasks import REPO_ROOT

BYOK_LIST_URL = "https://openrouter.ai/api/v1/byok"
CACHE_PATH = REPO_ROOT / ".cache" / "openrouter_byok.json"
CACHE_TTL_S = 3600
POLICIES = {
    "byok": "BYOK providers only",
    "byok_first": "BYOK first, then shared capacity",
    "routing": "OpenRouter routing",
    "pin": "Pinned provider",
}


def slug(tag_or_name: str) -> str:
    """'cerebras/fp16' -> 'cerebras'; 'Cerebras' -> 'cerebras'."""
    return tag_or_name.split("/", 1)[0].strip().lower()


def _from_env() -> set[str]:
    raw = os.getenv("OPENROUTER_BYOK_PROVIDERS", "")
    return {slug(x) for x in raw.replace(";", ",").split(",") if x.strip()}


def _from_api(timeout: int = 10) -> set[str] | None:
    key = os.getenv("OPENROUTER_MANAGEMENT_KEY")
    if not key:
        return None
    try:
        if CACHE_PATH.exists():
            cached = json.loads(CACHE_PATH.read_text())
            if time.time() - cached["fetched_at"] < CACHE_TTL_S:
                return set(cached["providers"])
        resp = requests.get(BYOK_LIST_URL, headers={"Authorization": f"Bearer {key}"},
                            params={"limit": 100}, timeout=timeout)
        resp.raise_for_status()
        providers = {
            slug(c["provider"]) for c in resp.json().get("data", [])
            if c.get("provider") and not c.get("disabled") and not c.get("is_fallback")
        }
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps({"fetched_at": time.time(), "providers": sorted(providers)}))
        return providers
    except Exception:  # noqa: BLE001 - optional source
        return None


def _from_runs(store) -> set[str]:
    found: set[str] = set()
    if store is None:
        return found
    for run in store.list(500):
        found.update(run.progress.get("byok_providers") or [])
    return found


def known_byok_providers(store=None) -> tuple[set[str], str]:
    """BYOK provider slugs and a short description of where they came from."""
    env = _from_env()
    if env:
        return env, "OPENROUTER_BYOK_PROVIDERS"
    api = _from_api()
    if api:
        return api, "OpenRouter BYOK keys (management key)"
    seen = _from_runs(store)
    if seen:
        return seen, "seen as BYOK in earlier runs"
    return set(), "none known"


def byok_tags(endpoints: ProviderList | None, byok: set[str]) -> list[str]:
    """Endpoint slugs of the BYOK providers that serve this model, healthy ones first."""
    if endpoints is None or not byok:
        return []
    serving = [p for p in endpoints.providers if slug(p.tag) in byok]
    serving.sort(key=lambda p: not p.healthy)
    return list(dict.fromkeys(slug(p.tag) for p in serving))


def provider_preferences(
    policy: str, pinned: str | None, endpoints: ProviderList | None, byok: set[str]
) -> tuple[dict[str, Any] | None, str]:
    """The OpenRouter ``provider`` request object for a policy, and a description of it.

    When no BYOK provider serves the model, BYOK policies fall back to OpenRouter routing.
    When the endpoint list is unavailable, the BYOK slugs are used as they are.
    """
    if policy == "pin" and pinned:
        return {"order": [pinned], "allow_fallbacks": False}, f"pinned to {pinned}"
    if policy in ("byok", "byok_first"):
        order = byok_tags(endpoints, byok) if endpoints and endpoints.providers else sorted(byok)
        if order:
            strict = policy == "byok"
            return (
                {"order": order, "allow_fallbacks": not strict},
                ("BYOK only: " if strict else "BYOK first: ") + ", ".join(order),
            )
        if byok:
            return None, f"no BYOK provider ({', '.join(sorted(byok))}) serves this model - OpenRouter routing"
        return None, "no BYOK provider known - OpenRouter routing"
    return None, "OpenRouter routing"
