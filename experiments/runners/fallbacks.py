"""Fallback runners when native dependencies are unavailable."""

from __future__ import annotations

import json
from typing import Iterable

import pandas as pd
from experiments.common.timing import timed


def _best_candidate(candidates: list[dict]) -> tuple[str | None, float | None]:
    if not candidates:
        return None, None
    best = candidates[0]
    score = best.get("scores", {}).get("es")
    return best.get("id"), score


def run_lamapi_top1(
    table_df: pd.DataFrame,
    mention_columns: Iterable[str],
    client,
    target_kg: str | None,
    forced_map: dict[str, list[str]] | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    forced_map = forced_map or {}

    for row_idx in range(len(table_df)):
        for col_name in mention_columns:
            col_idx = table_df.columns.get_loc(col_name)
            key = f"{row_idx}-{col_idx}"
            mention = table_df.iloc[row_idx, col_idx]
            if pd.isna(mention):
                rows.append(
                    {
                        "key": key,
                        "pred_id": "NIL",
                        "score": None,
                        "runtime_ms": None,
                        "meta_json": json.dumps({"reason": "empty_mention"}),
                    }
                )
                continue
            mention_text = str(mention)
            forced_ids = forced_map.get(key)
            with timed() as timer:
                candidates = client.fetch_entities(
                    mention_text,
                    ids=forced_ids,
                    target_kg=target_kg,
                )
            best_id, best_score = _best_candidate(candidates)
            pred_id = best_id or "NIL"
            rows.append(
                {
                    "key": key,
                    "pred_id": pred_id,
                    "score": best_score,
                    "runtime_ms": round(timer.elapsed_ms, 3),
                    "meta_json": json.dumps(
                        {
                            "mention": mention_text,
                            "best_id": best_id,
                            "best_score": best_score,
                            "forced_ids": forced_ids,
                        },
                        ensure_ascii=True,
                    ),
                }
            )

    return rows
