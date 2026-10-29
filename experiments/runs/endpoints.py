"""Saved LLM endpoints and their API keys, managed from the console's Endpoints page.

Keys live in ``secrets/endpoints.json`` (gitignored, owner-only permissions; override the path with
``ENDPOINTS_FILE``), never in a run's config, the run store or its output folder: a run records only
the endpoint's base URL and the worker looks the key up when the run starts. OpenRouter is always
available; its key falls back to ``OPENROUTER_API_KEY`` from the environment when none is saved.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.cost.tasks import REPO_ROOT
from experiments.runs.llm import normalize_endpoint

OPENROUTER = "OpenRouter"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"


@dataclass
class Endpoint:
    name: str
    base_url: str | None  # OpenAI-compatible base URL; None = OpenRouter
    api_key: str | None = None

    @property
    def is_openrouter(self) -> bool:
        return self.base_url is None


def endpoints_path() -> Path:
    value = os.getenv("ENDPOINTS_FILE")
    return Path(value).expanduser() if value else REPO_ROOT / "secrets" / "endpoints.json"


def _read() -> list[Endpoint]:
    path = endpoints_path()
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []
    return [Endpoint(e["name"], normalize_endpoint(e.get("base_url")), e.get("api_key") or None) for e in raw]


def _write(endpoints: list[Endpoint]) -> None:
    path = endpoints_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    # Created owner-only before the key is written to it.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump([asdict(e) for e in endpoints], fh, indent=2)
    os.replace(tmp, path)


def list_endpoints() -> list[Endpoint]:
    """OpenRouter first, then the saved custom endpoints by name."""
    saved = _read()
    openrouter = next((e for e in saved if e.is_openrouter), Endpoint(OPENROUTER, None))
    return [openrouter] + sorted((e for e in saved if not e.is_openrouter), key=lambda e: e.name.lower())


def get_endpoint(name_or_url: str | None) -> Endpoint | None:
    """By name, or by base URL (None or "" = OpenRouter)."""
    url = normalize_endpoint(name_or_url) if name_or_url and "://" in name_or_url else None
    for e in list_endpoints():
        if (not name_or_url and e.is_openrouter) or e.name == name_or_url or (url and e.base_url == url):
            return e
    return None


def save_endpoint(name: str, base_url: str | None, api_key: str | None = None, keep_key: bool = True,
                  old_name: str | None = None) -> Endpoint:
    """Add or update a custom endpoint. With ``keep_key`` an empty ``api_key`` keeps the saved one."""
    name = name.strip()
    base_url = normalize_endpoint(base_url)
    if not name:
        raise ValueError("The endpoint needs a name.")
    if not base_url:
        raise ValueError("The endpoint needs a base URL.")
    saved = _read()
    previous = next((e for e in saved if e.name == (old_name or name) and not e.is_openrouter), None)
    if name == OPENROUTER or any(e.name == name and e is not previous for e in saved):
        raise ValueError(f"An endpoint named '{name}' already exists.")
    if any(e.base_url == base_url and e is not previous for e in saved):
        raise ValueError(f"{base_url} is already saved under another name.")
    key = (api_key or "").strip() or (previous.api_key if previous and keep_key else None)
    endpoint = Endpoint(name, base_url, key)
    _write([e for e in saved if e is not previous] + [endpoint])
    return endpoint


def set_openrouter_key(api_key: str | None) -> None:
    """Save (or with None, forget) the OpenRouter key; without one ``OPENROUTER_API_KEY`` is used."""
    saved = [e for e in _read() if not e.is_openrouter]
    _write(saved + ([Endpoint(OPENROUTER, None, api_key.strip())] if api_key and api_key.strip() else []))


def delete_endpoint(name: str) -> None:
    _write([e for e in _read() if e.name != name or e.is_openrouter])


def api_key_for(base_url: str | None) -> str | None:
    """The key to send to this endpoint (None = OpenRouter); None when it needs none."""
    endpoint = get_endpoint(base_url)
    if endpoint and endpoint.api_key:
        return endpoint.api_key
    return os.getenv(OPENROUTER_KEY_ENV) if base_url is None else None


def key_source(endpoint: Endpoint) -> str:
    """Where the endpoint's key comes from, for display (never the key itself)."""
    if endpoint.api_key:
        return f"saved key ····{endpoint.api_key[-4:]}" if len(endpoint.api_key) > 8 else "saved key"
    if endpoint.is_openrouter and os.getenv(OPENROUTER_KEY_ENV):
        return f"{OPENROUTER_KEY_ENV} from the environment"
    return "no key"


def resolve_endpoint_url(name_or_url: str | None) -> str | None:
    """Command-line ``--endpoint``: a saved endpoint's name or a base URL -> base URL (None = OpenRouter)."""
    if not name_or_url or name_or_url == OPENROUTER:
        return None
    endpoint = get_endpoint(name_or_url)
    if endpoint:
        return endpoint.base_url
    if "://" not in name_or_url and "/" not in name_or_url and ":" not in name_or_url:
        raise ValueError(f"No saved endpoint named '{name_or_url}' (add it on the Endpoints page, or pass its URL).")
    return normalize_endpoint(name_or_url)
