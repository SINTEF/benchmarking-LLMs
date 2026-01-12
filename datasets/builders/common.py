"""Shared dataset builder helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from experiments.common.io import ensure_dir
from experiments.common.schemas import build_key


def normalize_identifier(value) -> str:
    if value is None:
        return "NIL"
    text = str(value).strip()
    if not text:
        return "NIL"
    lowered = text.lower()
    if lowered in {"nil", "none", "nan"}:
        return "NIL"
    if "wikidata.org/entity/" in lowered:
        return text.rsplit("/", 1)[-1]
    if "wikidata.org/wiki/" in lowered:
        return text.rsplit("/", 1)[-1]
    if "crunchbase.com/organization/" in lowered:
        slug = text.split("crunchbase.com/organization/", 1)[-1]
        return slug.strip("/")
    return text


def write_metadata(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def build_gt_rows(
    table_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    mention_columns: list[str],
    target_kg: str | None,
    table_name: str | None = None,
) -> list[dict[str, object]]:
    if "idRow" not in gt_df.columns or "idCol" not in gt_df.columns:
        raise ValueError("GT is missing idRow/idCol columns.")

    entity_col = None
    for candidate in ["gt_id", "entity", "wikidata_qid", "crunchbase_id", "qid", "id"]:
        if candidate in gt_df.columns:
            entity_col = candidate
            break
    if entity_col is None:
        raise ValueError("GT is missing a known entity column.")

    rows: list[dict[str, object]] = []
    table_columns = list(table_df.columns)
    for _, row in gt_df.iterrows():
        id_row = int(row["idRow"])
        id_col = int(row["idCol"])
        if id_row >= len(table_df) or id_col >= len(table_columns):
            continue
        mention_col_name = table_columns[id_col]
        if mention_col_name not in mention_columns:
            continue
        mention_value = table_df.iloc[id_row, id_col]
        key = build_key(id_row, id_col)
        rows.append(
            {
                "key": key,
                "row": id_row,
                "col": id_col,
                "mention": "" if pd.isna(mention_value) else str(mention_value),
                "target_kg": target_kg or "",
                "gt_id": normalize_identifier(row[entity_col]),
            }
        )
    return rows


def filter_gt_to_table(gt_df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    if "tableName" not in gt_df.columns:
        return gt_df
    return gt_df[gt_df["tableName"] == table_name].copy()


def ensure_dir_for_file(path: Path) -> None:
    ensure_dir(path.parent)
