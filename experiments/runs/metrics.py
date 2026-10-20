"""Precision / recall / F1 of a run, final or while it is still going."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.cost.tasks import DatasetSource
from experiments.evaluation.metrics import evaluate
from experiments.forcing import load_gt_ids


def gt_frame(source: DatasetSource) -> pd.DataFrame | None:
    """GT as ``key, gt_id`` rows keyed like the executor's predictions (several rows per key allowed)."""
    from experiments.runs.executor import cell_key

    if source.kind == "spec":
        if not source.gt_path or not Path(source.gt_path).exists():
            return None
        df = pd.read_csv(source.gt_path)
        if "key" not in df.columns:
            df["key"] = df["row"].astype(int).astype(str) + "-" + df["col"].astype(int).astype(str)
        return df[["key", "gt_id"]]  # keeps NIL rows: they count for NIL detection and precision
    rows = [
        {"key": cell_key(t, r, c), "gt_id": gid}
        for (t, r, c), ids in load_gt_ids(source).items()
        for gid in ids
    ]
    return pd.DataFrame(rows, columns=["key", "gt_id"]) if rows else None


def load_results(run_dir: str | Path) -> pd.DataFrame:
    """Latest result per cell from ``results.jsonl`` (columns key, pred_id, status, ...)."""
    path = Path(run_dir) / "results.jsonl"
    latest: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                latest[row["key"]] = row
    return pd.DataFrame(latest.values())


def score(pred_df: pd.DataFrame, gt_df: pd.DataFrame | None) -> dict[str, Any] | None:
    """``evaluate`` restricted to predicted cells, or None when nothing can be scored yet."""
    if gt_df is None or pred_df.empty:
        return None
    gt_df = gt_df[gt_df["key"].isin(pred_df["key"])]
    if gt_df.empty:
        return None
    return evaluate(pred_df[["key", "pred_id"]], gt_df)


def run_metrics(run_dir: str | Path, source: DatasetSource) -> dict[str, Any] | None:
    """Metrics over the cells finished so far (cells that failed count as a NIL prediction)."""
    results = load_results(run_dir)
    if results.empty:
        return None
    return score(results, gt_frame(source))


def earlier_summary(source: DatasetSource, output_dir: str | Path = "outputs") -> str | None:
    """A table summary an earlier LLM run of this dataset wrote (the most recent), if any."""
    from experiments.cost.tasks import REPO_ROOT

    base = Path(output_dir)
    base = base if base.is_absolute() else REPO_ROOT / base
    files = sorted((base / source.name / "llm").glob("*/summaries.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files:
        try:
            values = [v for v in json.loads(f.read_text()).values() if v]
        except (OSError, json.JSONDecodeError):
            continue
        if values:
            return values[0]
    return None
