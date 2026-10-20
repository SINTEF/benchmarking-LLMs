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


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def compute_accuracy(pred_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict[str, Any]:
    """Accuracy plus entity-linking precision / recall / F1 over the cells that have a GT row.

    A cell may have several GT rows (alternative correct ids): a prediction matching any of them
    is correct. Definitions (as in SemTab CEA, NIL = "no entity"):

    * precision = correct entity predictions / cells where an entity (non-NIL) was predicted
    * recall    = correct entity predictions / cells whose GT is an entity
    * F1        = harmonic mean of the two
    * accuracy  = cells right, NIL included / all evaluated cells
    * nil_*     = the same three for NIL detection (predicting NIL where the GT is NIL)
    """
    gt = gt_df[["key", "gt_id"]].copy()
    gt["gt_norm"] = gt["gt_id"].apply(normalize_identifier)
    gold = gt.groupby("key")["gt_norm"].agg(lambda ids: frozenset(ids)).rename("gold")
    pred = pred_df.drop_duplicates("key", keep="last").set_index("key")["pred_id"].apply(normalize_identifier)
    merged = pd.concat([pred.rename("pred_norm"), gold], axis=1, join="inner")
    if merged.empty:
        raise ValueError("No overlapping keys between predictions and GT.")

    gt_nil = merged["gold"].apply(lambda ids: ids == frozenset({"NIL"}))
    pred_nil = merged["pred_norm"] == "NIL"
    hit = merged.apply(lambda r: r["pred_norm"] in r["gold"], axis=1)

    evaluated = int(len(merged))
    correct = int(hit.sum())
    tp = int((hit & ~pred_nil).sum())
    predicted = int((~pred_nil).sum())
    gold_entities = int((~gt_nil).sum())
    nil_correct = int((gt_nil & pred_nil).sum())
    precision, recall = _ratio(tp, predicted), _ratio(tp, gold_entities)
    nil_precision, nil_recall = _ratio(nil_correct, int(pred_nil.sum())), _ratio(nil_correct, int(gt_nil.sum()))

    return {
        "evaluated": evaluated,
        "correct": correct,
        "accuracy": _ratio(correct, evaluated),
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "tp": tp,
        "predicted_entities": predicted,
        "gold_entities": gold_entities,
        "nil_gt_count": int(gt_nil.sum()),
        "nil_pred_count": int(pred_nil.sum()),
        "nil_correct": nil_correct,
        "nil_precision": nil_precision,
        "nil_recall": nil_recall,
        "nil_f1": _f1(nil_precision, nil_recall),
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
