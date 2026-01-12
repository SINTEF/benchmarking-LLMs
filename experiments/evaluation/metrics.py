"""Evaluation metrics for canonical predictions."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


def normalize_identifier(value: Any) -> str:
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


def compute_accuracy(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict[str, Any]:
    merged = pred_df.merge(gt_df, on="key", how="inner", suffixes=("_pred", "_gt"))
    if merged.empty:
        raise ValueError("No overlapping keys between predictions and GT.")

    merged["pred_norm"] = merged["pred_id"].apply(normalize_identifier)
    merged["gt_norm"] = merged["gt_id"].apply(normalize_identifier)
    correct_mask = merged["pred_norm"] == merged["gt_norm"]

    evaluated = int(len(merged))
    correct = int(correct_mask.sum())
    accuracy = correct / evaluated if evaluated else 0.0
    nil_gt_count = int((merged["gt_norm"] == "NIL").sum())
    nil_pred_count = int((merged["pred_norm"] == "NIL").sum())
    nil_correct = int(((merged["gt_norm"] == "NIL") & (merged["pred_norm"] == "NIL")).sum())

    return {
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": accuracy,
        "nil_gt_count": nil_gt_count,
        "nil_pred_count": nil_pred_count,
        "nil_correct": nil_correct,
    }


def summarize_runtime(pred_df: pd.DataFrame) -> dict[str, Any] | None:
    if "runtime_ms" not in pred_df.columns:
        return None
    runtime = pd.to_numeric(pred_df["runtime_ms"], errors="coerce").dropna()
    if runtime.empty:
        return None
    return {
        "count": int(runtime.count()),
        "mean_ms": float(runtime.mean()),
        "median_ms": float(runtime.median()),
        "p95_ms": float(runtime.quantile(0.95)),
        "max_ms": float(runtime.max()),
    }


def evaluate(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict[str, Any]:
    accuracy = compute_accuracy(pred_df, gt_df)
    runtime = summarize_runtime(pred_df)
    report = {
        "total_pred": int(len(pred_df)),
        "total_gt": int(len(gt_df)),
    }
    report.update(accuracy)
    if runtime:
        report["runtime_ms"] = runtime
    return report
