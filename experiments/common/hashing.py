"""Stable hashing helpers."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_hash(payload: dict[str, Any], length: int = 12) -> str:
    """Return a stable short hash for a JSON-serializable payload."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return digest[:length]
