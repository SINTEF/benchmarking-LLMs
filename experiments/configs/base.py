"""Dataclasses for runner configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LamapiConfig:
    endpoint: str
    token: str | None = None
    limit: int = 10
    cache_dir: Path | None = None
    offline: bool = False


@dataclass
class ForcingConfig:
    force_gt_candidate: bool = False
    force_ids: list[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    name: str
    table_path: Path
    gt_path: Path | None
    mention_columns: list[str]
    target_kg: str | None


@dataclass
class OutputConfig:
    base_dir: Path = Path("outputs")
    method: str = ""


@dataclass
class RunnerConfig:
    dataset: DatasetConfig
    lamapi: LamapiConfig
    forcing: ForcingConfig
    output: OutputConfig
    max_rows: int = 0
    extra: dict[str, Any] = field(default_factory=dict)
