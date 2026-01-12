"""LamAPI retrieval wrapper with deterministic caching."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import requests

from experiments.common.logging import get_logger

logger = get_logger(__name__)


def _normalize_ids(ids: str | Iterable[str] | None) -> list[str] | None:
    if ids is None:
        return None
    if isinstance(ids, str):
        raw = [ids]
    else:
        raw = list(ids)
    cleaned = [str(item).strip() for item in raw if str(item).strip()]
    if not cleaned:
        return None
    return sorted(set(cleaned))


def _cache_key(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _safe_json_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def _extract_candidates(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("candidates", "entities", "results", "data"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        # Some LamAPI variants return {mention: [candidates]}
        if len(payload) == 1:
            value = next(iter(payload.values()))
            if isinstance(value, list):
                return value
    return []


def _normalize_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, float] = {}
    for key, value in candidate.items():
        if value is None:
            continue
        if key.endswith("_score"):
            base = key[: -len("_score")]
            try:
                scores[base] = float(value)
            except (TypeError, ValueError):
                continue
        if key == "ed":
            try:
                scores["ed"] = float(value)
            except (TypeError, ValueError):
                continue

    label = candidate.get("label") or candidate.get("name")
    if label is None and "title" in candidate:
        label = candidate.get("title")

    aliases = candidate.get("aliases")
    if aliases is None and "alias" in candidate:
        aliases = candidate.get("alias")

    types = candidate.get("types")
    return {
        "id": candidate.get("id") or candidate.get("qid") or candidate.get("entity_id"),
        "label": label,
        "description": candidate.get("description"),
        "types": types if types is not None else [],
        "aliases": aliases if aliases is not None else [],
        "scores": scores,
        "raw": candidate,
    }


def normalize_candidates(payload: Any) -> list[dict[str, Any]]:
    return [_normalize_candidate(candidate) for candidate in _extract_candidates(payload)]


class LamapiClient:
    """Simple LamAPI HTTP client with deterministic caching."""

    def __init__(
        self,
        endpoint: str,
        token: str | None = None,
        num_candidates: int = 10,
        cache_dir: str | Path | None = None,
        offline: bool = False,
        query_param: str | None = None,
        token_in_query: bool | None = None,
        token_param: str | None = None,
    ) -> None:
        if not endpoint:
            raise ValueError("LamAPI endpoint is required.")
        self.endpoint = endpoint
        self.token = token
        self.num_candidates = num_candidates
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/lamapi_cache")
        self.offline = offline
        self.query_param = query_param or os.getenv("LAMAPI_QUERY_PARAM", "name")
        self.token_in_query = (
            token_in_query
            if token_in_query is not None
            else os.getenv("LAMAPI_TOKEN_IN_QUERY", "1") == "1"
        )
        self.token_param = token_param or os.getenv("LAMAPI_TOKEN_PARAM", "token")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _request_payload(
        self,
        name: str,
        limit: int,
        ids: list[str] | None,
        target_kg: str | None,
    ) -> dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "name": name,
            "limit": limit,
            "ids": ids,
            "target_kg": target_kg,
        }

    def _cache_lookup(self, payload: dict[str, Any]) -> tuple[str, Path | None, Any | None]:
        key = _cache_key(payload)
        path = _safe_json_path(self.cache_dir, key)
        if path.exists():
            return key, path, json.loads(path.read_text(encoding="utf-8"))
        return key, path, None

    def _cache_store(self, path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _fetch_raw(
        self,
        name: str,
        ids: str | Iterable[str] | None = None,
        target_kg: str | None = None,
    ) -> Any:
        ids_list = _normalize_ids(ids)
        payload = self._request_payload(name, self.num_candidates, ids_list, target_kg)
        key, path, cached = self._cache_lookup(payload)
        if cached is not None:
            return cached
        if self.offline:
            raise RuntimeError(f"LamAPI offline mode: cache miss for {name}")

        params: dict[str, Any] = {self.query_param: name, "limit": self.num_candidates}
        if ids_list:
            params["ids"] = ",".join(ids_list)
        if target_kg:
            params["target_kg"] = target_kg
        if self.token and self.token_in_query:
            params[self.token_param] = self.token

        headers = {}
        if self.token and not self.token_in_query:
            headers["Authorization"] = f"Bearer {self.token}"

        logger.debug("LamAPI request %s params=%s", self.endpoint, params)
        response = requests.get(self.endpoint, params=params, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json()
        self._cache_store(path, data)
        logger.debug("LamAPI cached response %s", key)
        return data

    def fetch_entities(
        self,
        name: str,
        ids: str | Iterable[str] | None = None,
        target_kg: str | None = None,
    ) -> list[dict[str, Any]]:
        raw = self._fetch_raw(name, ids=ids, target_kg=target_kg)
        return normalize_candidates(raw)

    def fetch_multiple_entities(
        self,
        mentions: Iterable[str],
        mention_to_qids: dict[str, str | list[str]] | None = None,
        target_kg: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        output: dict[str, list[dict[str, Any]]] = {}
        for mention in mentions:
            forced_ids = None
            if mention_to_qids:
                forced_ids = mention_to_qids.get(mention)
            output[mention] = self.fetch_entities(mention, ids=forced_ids, target_kg=target_kg)
        return output
