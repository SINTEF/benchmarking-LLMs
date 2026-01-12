"""Dataset spec registry loaded from JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    description: str
    source_table: str
    source_gt: str
    table_path: str
    gt_path: str
    mention_columns: list[str]
    target_kg: str | None
    seed: int
    table_columns: list[str]


_SPECS_DIR = Path(__file__).resolve().parent


def _load_spec_file(path: Path) -> DatasetSpec:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DatasetSpec(**payload)


def list_specs() -> dict[str, DatasetSpec]:
    specs: dict[str, DatasetSpec] = {}
    for path in sorted(_SPECS_DIR.glob("*.json")):
        spec = _load_spec_file(path)
        specs[spec.name] = spec
    return specs


def get_spec(name: str) -> DatasetSpec:
    specs = list_specs()
    if name not in specs:
        available = ", ".join(sorted(specs))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return specs[name]
