"""Lightweight SPARQL helpers for dataset builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests


def load_query_template(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def run_sparql_query(endpoint: str, query: str, timeout: int = 60) -> dict[str, Any]:
    headers = {"Accept": "application/sparql-results+json"}
    response = requests.get(endpoint, params={"query": query, "format": "json"}, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()
