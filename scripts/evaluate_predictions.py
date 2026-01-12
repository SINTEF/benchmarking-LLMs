#!/usr/bin/env python3
"""Evaluate LionLinker predictions against ground truth."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import pandas as pd


GT_COL_CANDIDATES = ["gt_id", "entity", "wikidata_qid", "crunchbase_id", "qid", "id"]
PRED_COL_FALLBACK_SUFFIX = "_pred_id"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions against GT.")
    parser.add_argument("--pred_csv", required=True, type=str)
    parser.add_argument("--gt_csv", required=True, type=str)
    parser.add_argument("--id_col", default="id", type=str)
    parser.add_argument("--pred_col", default="predicted_id", type=str)
    parser.add_argument("--gt_col", default="gt_id", type=str)
    parser.add_argument("--report_path", default=None, type=str)
    return parser.parse_args()


def normalize_identifier(value) -> str:
    if value is None:
        return "NIL"
    if isinstance(value, float) and math.isnan(value):
        return "NIL"
    text = str(value).strip()
    if not text:
        return "NIL"
    lowered = text.lower()
    if lowered in {"nil", "none", "no identifier", "nan"}:
        return "NIL"
    if "wikidata.org/entity/" in lowered:
        return text.rsplit("/", 1)[-1]
    if "wikidata.org/wiki/" in lowered:
        return text.rsplit("/", 1)[-1]
    if "crunchbase.com/organization/" in lowered:
        slug = text.split("crunchbase.com/organization/", 1)[-1]
        return slug.strip("/")
    return text


def resolve_gt_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    for candidate in GT_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Ground truth column not found. Tried '{requested}' "
        f"and {', '.join(GT_COL_CANDIDATES)}."
    )


def resolve_pred_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    pred_cols = [col for col in df.columns if col.endswith(PRED_COL_FALLBACK_SUFFIX)]
    if len(pred_cols) == 1:
        return pred_cols[0]
    raise ValueError(
        f"Prediction column not found. Tried '{requested}' and '*{PRED_COL_FALLBACK_SUFFIX}'."
    )


def parse_table_from_id(value: str) -> Optional[str]:
    parts = str(value).split("-")
    if len(parts) < 3:
        return None
    return "-".join(parts[:-2])


def filter_gt_to_test(gt_df: pd.DataFrame, pred_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if "tableName" not in gt_df.columns:
        return gt_df
    if "tableName" in pred_df.columns:
        tables = set(pred_df["tableName"].dropna().unique())
        return gt_df[gt_df["tableName"].isin(tables)]
    if id_col in pred_df.columns:
        tables = set()
        for value in pred_df[id_col].dropna().unique():
            table = parse_table_from_id(value)
            if table:
                tables.add(table)
        if tables:
            return gt_df[gt_df["tableName"].isin(tables)]
    return gt_df[~gt_df["tableName"].str.contains("train", case=False, na=False)]


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred_csv)
    gt_path = Path(args.gt_csv)

    pred_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)

    id_col = args.id_col
    if id_col not in pred_df.columns:
        pred_df = pred_df.copy()
        if {"tableName", "idRow", "idCol"}.issubset(pred_df.columns):
            pred_df[id_col] = (
                pred_df["tableName"].astype(str)
                + "-"
                + pred_df["idRow"].astype(str)
                + "-"
                + pred_df["idCol"].astype(str)
            )
        else:
            pred_df[id_col] = pred_df.index.astype(str)

    if id_col not in gt_df.columns:
        if {"tableName", "idRow", "idCol"}.issubset(gt_df.columns):
            gt_df = gt_df.copy()
            gt_df[id_col] = (
                gt_df["tableName"].astype(str)
                + "-"
                + gt_df["idRow"].astype(str)
                + "-"
                + gt_df["idCol"].astype(str)
            )
        else:
            raise ValueError(
                f"GT is missing '{id_col}' and tableName/idRow/idCol columns "
                f"to construct it. Available columns: {', '.join(gt_df.columns)}"
            )

    pred_col = resolve_pred_column(pred_df, args.pred_col)
    gt_col = resolve_gt_column(gt_df, args.gt_col)

    gt_df = filter_gt_to_test(gt_df, pred_df, id_col)

    merged = pred_df.merge(gt_df, on=id_col, how="inner", suffixes=("_pred", "_gt"))
    if merged.empty:
        raise ValueError("No overlapping rows between predictions and GT after filtering.")

    merged["pred_norm"] = merged[pred_col].apply(normalize_identifier)
    merged["gt_norm"] = merged[gt_col].apply(normalize_identifier)

    correct_mask = merged["pred_norm"] == merged["gt_norm"]
    evaluated = len(merged)
    correct = int(correct_mask.sum())
    accuracy = correct / evaluated if evaluated else 0.0

    nil_gt_count = int((merged["gt_norm"] == "NIL").sum())
    nil_pred_count = int((merged["pred_norm"] == "NIL").sum())
    nil_correct = int(((merged["gt_norm"] == "NIL") & (merged["pred_norm"] == "NIL")).sum())

    report = {
        "total_gt": int(len(gt_df)),
        "total_pred": int(len(pred_df)),
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": accuracy,
        "nil_gt_count": nil_gt_count,
        "nil_pred_count": nil_pred_count,
        "nil_correct": nil_correct,
    }

    report_path = Path(args.report_path) if args.report_path else pred_path.parent / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    mismatches = merged.loc[~correct_mask].copy()
    mismatch_cols = [id_col, pred_col, gt_col]
    if "mention" in merged.columns:
        mismatch_cols.insert(1, "mention")
    mismatches = mismatches[mismatch_cols].head(20)
    mismatches_path = pred_path.parent / "mismatches.csv"
    mismatches.to_csv(mismatches_path, index=False)

    print(
        "Accuracy: "
        f"{accuracy:.4f} ({correct}/{evaluated}). "
        f"NIL gt/pred/correct: {nil_gt_count}/{nil_pred_count}/{nil_correct}."
    )


if __name__ == "__main__":
    main()
