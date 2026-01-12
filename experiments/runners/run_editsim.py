#!/usr/bin/env python3
"""Run the LamAPI edit-sim baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

from experiments.common.io import ensure_dir, write_json
from experiments.common.env import load_dotenv
from experiments.common.logging import get_logger
from experiments.common.paths import resolve_run_dir
from experiments.common.schemas import PRED_COLUMNS
from experiments.common.timing import timed
from experiments.forcing import build_correct_qids_from_gt, merge_forced_ids
from experiments.retrieval.lamapi_client import LamapiClient
from experiments.runners.common import (
    add_common_args,
    add_dataset_args,
    build_table_keys,
    resolve_mention_columns,
    resolve_paths,
    resolve_spec,
    resolve_target_kg,
)

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LamAPI edit similarity baseline.")
    add_dataset_args(parser)
    add_common_args(parser)
    parser.add_argument("--nil-threshold", type=float, default=0.0)
    return parser.parse_args()


def _best_ed_candidate(candidates: list[dict]) -> tuple[str | None, float | None]:
    best_id = None
    best_score = None
    for candidate in candidates:
        score = candidate.get("scores", {}).get("ed")
        if score is None:
            continue
        if best_score is None or score > best_score:
            best_score = score
            best_id = candidate.get("id")
    return best_id, best_score


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.max_rows < 0:
        raise ValueError("--max-rows must be >= 0")

    spec = resolve_spec(args)
    table_path, gt_path = resolve_paths(args, spec)
    target_kg = resolve_target_kg(args, spec)

    table_df = pd.read_csv(table_path)
    mention_columns = resolve_mention_columns(table_df, spec.mention_columns)

    if args.max_rows:
        table_df = table_df.head(args.max_rows)

    keys = build_table_keys(table_df, mention_columns)
    key_set = set(keys)

    gt_forcing = {}
    if args.force_gt_candidate:
        if not gt_path:
            raise ValueError("--force-gt-candidate requires --gt-path or dataset GT.")
        gt_forcing = build_correct_qids_from_gt(gt_path)
        gt_forcing = {key: value for key, value in gt_forcing.items() if key in key_set}

    forced_map = merge_forced_ids(gt_forcing, args.force_id, keys=keys)

    lamapi_endpoint = args.lamapi_endpoint
    if not lamapi_endpoint:
        raise RuntimeError("Missing --lamapi-endpoint or ENTITY_RETRIEVAL_ENDPOINT.")
    lamapi_cache_dir = args.lamapi_cache_dir or "data/lamapi_cache"
    offline = args.lamapi_offline or os.getenv("LAMAPI_OFFLINE", "0") == "1"
    client = LamapiClient(
        endpoint=lamapi_endpoint,
        token=args.lamapi_token,
        num_candidates=args.lamapi_limit,
        cache_dir=lamapi_cache_dir,
        offline=offline,
    )

    settings = {
        "dataset": spec.name,
        "method": "editsim",
        "max_rows": args.max_rows,
        "nil_threshold": args.nil_threshold,
        "lamapi_limit": args.lamapi_limit,
        "force_gt_candidate": args.force_gt_candidate,
        "force_id": args.force_id,
        "target_kg": target_kg,
    }
    run_dir = resolve_run_dir(Path(args.output_dir), spec.name, "editsim", settings)
    ensure_dir(run_dir)

    rows = []
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
            best_id, best_score = _best_ed_candidate(candidates)
            pred_id = "NIL"
            if best_id and best_score is not None and best_score >= args.nil_threshold:
                pred_id = best_id
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

    pred_path = run_dir / "predictions.csv"
    pd.DataFrame(rows, columns=PRED_COLUMNS).to_csv(pred_path, index=False)
    write_json(run_dir / "manifest.json", settings)
    logger.info("Wrote predictions to %s", pred_path)


if __name__ == "__main__":
    main()
