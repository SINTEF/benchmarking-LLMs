"""Shared runner utilities."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import pandas as pd

from datasets.specs.registry import DatasetSpec, get_spec, list_specs
from experiments.common.logging import get_logger
from experiments.common.schemas import build_key
from experiments.retrieval.lamapi_client import resolve_lamapi_endpoint, resolve_lamapi_token

logger = get_logger(__name__)


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    specs = list_specs()
    parser.add_argument("--dataset", choices=sorted(specs.keys()), required=True)
    parser.add_argument("--table-path", type=str, default=None)
    parser.add_argument("--gt-path", type=str, default=None)
    parser.add_argument("--target-kg", type=str, default=None)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--force-gt-candidate", action="store_true")
    parser.add_argument("--force-id", action="append", default=[])
    parser.add_argument("--lamapi-endpoint", type=str, default=resolve_lamapi_endpoint())
    parser.add_argument("--lamapi-token", type=str, default=resolve_lamapi_token())
    parser.add_argument("--lamapi-limit", type=int, default=10)
    parser.add_argument("--lamapi-cache-dir", type=str, default=os.getenv("LAMAPI_CACHE_DIR"))
    parser.add_argument("--lamapi-offline", action="store_true")


def resolve_spec(args: argparse.Namespace) -> DatasetSpec:
    return get_spec(args.dataset)


def resolve_paths(args: argparse.Namespace, spec: DatasetSpec) -> tuple[Path, Path | None]:
    table_path = Path(args.table_path) if args.table_path else Path(spec.table_path)
    gt_path = Path(args.gt_path) if args.gt_path else Path(spec.gt_path)
    if not table_path.exists():
        raise FileNotFoundError(f"Missing table: {table_path}")
    if gt_path and not gt_path.exists():
        raise FileNotFoundError(f"Missing GT: {gt_path}")
    return table_path, gt_path


def resolve_target_kg(args: argparse.Namespace, spec: DatasetSpec) -> str | None:
    return args.target_kg or spec.target_kg


def resolve_mention_columns(df: pd.DataFrame, mention_columns: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for name in mention_columns:
        if name in df.columns:
            resolved.append(name)
            continue
        matches = [col for col in df.columns if col.lower() == name.lower()]
        if len(matches) == 1:
            resolved.append(matches[0])
        else:
            raise ValueError(f"Mention column '{name}' not found in table columns.")
    return resolved


def build_table_keys(df: pd.DataFrame, mention_columns: Iterable[str]) -> list[str]:
    keys: list[str] = []
    column_positions = [df.columns.get_loc(col) for col in mention_columns]
    for row_idx in range(len(df)):
        for col_idx in column_positions:
            keys.append(build_key(row_idx, col_idx))
    return keys
