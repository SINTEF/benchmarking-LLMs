"""Canonical schemas for GT and predictions."""

from __future__ import annotations

GT_COLUMNS = ["key", "row", "col", "mention", "target_kg", "gt_id"]
PRED_COLUMNS = ["key", "pred_id", "score", "runtime_ms", "meta_json"]


def build_key(row: int, col: int) -> str:
    return f"{row}-{col}"
