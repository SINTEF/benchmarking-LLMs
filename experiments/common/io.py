"""Shared IO utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_table(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if max_rows and max_rows > 0:
        return df.head(max_rows)
    return df


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    df = pd.DataFrame(list(rows))
    if columns:
        for col in columns:
            if col not in df.columns:
                df[col] = None
        df = df[columns]
    df.to_csv(path, index=False)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
