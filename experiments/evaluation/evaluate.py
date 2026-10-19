#!/usr/bin/env python3
"""Evaluate predictions against canonical GT."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.common.io import write_json
from experiments.common.schemas import build_key
from experiments.evaluation.metrics import evaluate, normalize_identifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate canonical predictions.")
    parser.add_argument("--pred", required=True, type=str, help="Path to predictions.csv")
    parser.add_argument("--gt", required=True, type=str, help="Path to gt.csv")
    parser.add_argument("--report", default=None, type=str, help="Path for report.json")
    parser.add_argument("--mismatches", default=None, type=str, help="Path for mismatches.csv")
    return parser.parse_args()


def _ensure_key(df: pd.DataFrame) -> pd.DataFrame:
    if "key" in df.columns:
        return df
    df = df.copy()
    if {"row", "col"}.issubset(df.columns):
        df["key"] = df.apply(lambda row: build_key(int(row["row"]), int(row["col"])), axis=1)
        return df
    if {"idRow", "idCol"}.issubset(df.columns):
        df["key"] = df.apply(lambda row: build_key(int(row["idRow"]), int(row["idCol"])), axis=1)
        return df
    raise ValueError("Missing key column and row/col or idRow/idCol columns.")


def main() -> None:
    args = parse_args()
    pred_path = Path(args.pred)
    gt_path = Path(args.gt)

    pred_df = pd.read_csv(pred_path)
    gt_df = pd.read_csv(gt_path)

    pred_df = _ensure_key(pred_df)
    gt_df = _ensure_key(gt_df)

    report = evaluate(pred_df, gt_df)
    report_path = Path(args.report) if args.report else pred_path.parent / "report.json"
    write_json(report_path, report)

    mismatches_path = Path(args.mismatches) if args.mismatches else pred_path.parent / "mismatches.csv"
    merged = pred_df.merge(gt_df, on="key", how="inner", suffixes=("_pred", "_gt"))
    merged["pred_norm"] = merged["pred_id"].apply(normalize_identifier)
    merged["gt_norm"] = merged["gt_id"].apply(normalize_identifier)
    mismatches = merged.loc[merged["pred_norm"] != merged["gt_norm"]]
    cols = ["key", "pred_id", "gt_id"]
    if "mention" in merged.columns:
        cols.insert(1, "mention")
    mismatches[cols].head(50).to_csv(mismatches_path, index=False)

    print(
        f"P {report['precision']:.4f}  R {report['recall']:.4f}  F1 {report['f1']:.4f}  "
        f"accuracy {report['accuracy']:.4f} ({report['correct']}/{report['evaluated']})."
    )


if __name__ == "__main__":
    main()
