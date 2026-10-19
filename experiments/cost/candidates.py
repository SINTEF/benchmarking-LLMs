"""LamAPI candidate lookup for estimation, sharing the runners' on-disk cache."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

from experiments.cost.tasks import REPO_ROOT


def make_lamapi_clients(lamapi_limit: int, cache_dir: str | Path | None = None):
    """(offline_client, online_client) sharing the runners' cache (same endpoint -> same cache keys)."""
    from experiments.retrieval.lamapi_client import LamapiClient, resolve_lamapi_endpoint, resolve_lamapi_token

    endpoint = resolve_lamapi_endpoint()
    cache = cache_dir or os.getenv("LAMAPI_CACHE_DIR") or REPO_ROOT / "data" / "lamapi_cache"
    token = resolve_lamapi_token()
    kwargs = dict(endpoint=endpoint, token=token, num_candidates=lamapi_limit, cache_dir=cache)
    return LamapiClient(offline=True, **kwargs), LamapiClient(offline=False, **kwargs)


def _plain(mention: Any) -> Any:
    # numpy scalar -> python, so the cache key is JSON-serialisable (same key the runner computes)
    return mention.item() if hasattr(mention, "item") else mention


def check_lamapi(timeout: int = 10) -> tuple[bool, str]:
    """Quick reachability check: one lookup for a well-known name."""
    import requests

    from experiments.retrieval.lamapi_client import resolve_lamapi_endpoint, resolve_lamapi_token

    url = resolve_lamapi_endpoint()
    try:
        r = requests.get(url, params={"name": "Oslo", "limit": 1, "token": resolve_lamapi_token()}, timeout=timeout)
        r.raise_for_status()
        return True, url
    except Exception as exc:
        return False, f"{url}: {exc}"


def lookup(client, mention: Any, target_kg: str | None, ids: tuple[str, ...] | None = None) -> list[dict] | None:
    if client is None:
        return None
    try:
        return client.fetch_entities(_plain(mention), ids=list(ids) if ids else None, target_kg=target_kg)
    except Exception:
        return None


Lookup = tuple[Any, tuple[str, ...] | None]  # (mention, forced ids) - one LamAPI request


def forced_ids_for(gt_ids: dict, table_id: str, row: int, col: int, force_gt: bool) -> tuple[str, ...] | None:
    """Ids a run forces into this cell's candidates (same rule as experiments.runs.executor)."""
    if not force_gt:
        return None
    ids = gt_ids.get((table_id, int(row), int(col)))
    return tuple(sorted(set(ids))) if ids else None


def resolve_candidates(
    lookups: Iterable[Lookup | Any],
    target_kg: str | None,
    clients: tuple[Any, Any],
    fetch_missing: bool,
    workers: int = 16,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[Lookup, list[dict] | None], dict[str, int]]:
    """(mention, forced ids) -> candidates (None = unavailable). Cache first, then LamAPI in parallel.

    A bare mention is the same as ``(mention, None)``.
    """
    offline, online = clients
    unique = list(dict.fromkeys(x if isinstance(x, tuple) else (x, None) for x in lookups))
    found: dict[Lookup, list[dict] | None] = {}
    todo = []
    for m, ids in unique:
        c = lookup(offline, m, target_kg, ids)
        found[(m, ids)] = c
        if c is None:
            todo.append((m, ids))
    stats = {"cached": len(unique) - len(todo), "fetched": 0, "failed": 0}
    if fetch_missing and online is not None and todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(lambda x: lookup(online, x[0], target_kg, x[1]), todo)
            for i, (k, c) in enumerate(zip(todo, results)):
                found[k] = c
                stats["fetched" if c is not None else "failed"] += 1
                if progress and i % 10 == 0:
                    progress(i + 1, len(todo))
    else:
        stats["failed"] = len(todo)
    return found, stats
