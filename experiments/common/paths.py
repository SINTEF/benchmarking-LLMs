"""Path helpers for outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hashing import stable_hash


def resolve_run_dir(base_dir: Path, dataset: str, method: str, settings: dict[str, Any]) -> Path:
    settings_hash = stable_hash(settings)
    run_dir = base_dir / dataset / method / settings_hash
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
