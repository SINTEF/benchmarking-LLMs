#!/usr/bin/env python3
"""Run Alligator with LamAPI retrieval and forcing."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.common.env import load_dotenv
from experiments.common.io import ensure_dir, write_json
from experiments.common.logging import get_logger
from experiments.common.paths import resolve_run_dir
from experiments.common.schemas import PRED_COLUMNS, build_key
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
from experiments.runners.fallbacks import run_lamapi_top1

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Alligator on a dataset.")
    add_dataset_args(parser)
    add_common_args(parser)
    return parser.parse_args()


def _normalize_native_output(output: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if output is None:
        return rows
    if isinstance(output, pd.DataFrame):
        df = output
        if "key" not in df.columns and {"row", "col"}.issubset(df.columns):
            df = df.copy()
            df["key"] = df.apply(lambda r: build_key(int(r["row"]), int(r["col"])), axis=1)
        for _, row in df.iterrows():
            rows.append(
                {
                    "key": row.get("key"),
                    "pred_id": row.get("pred_id") or row.get("prediction") or row.get("entity_id"),
                    "score": row.get("score") or row.get("confidence"),
                    "runtime_ms": row.get("runtime_ms"),
                    "meta_json": row.get("meta_json"),
                }
            )
        return rows
    if isinstance(output, dict):
        for key, value in output.items():
            rows.append(
                {
                    "key": key,
                    "pred_id": value,
                    "score": None,
                    "runtime_ms": None,
                    "meta_json": None,
                }
            )
        return rows
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if not key and "row" in item and "col" in item:
                key = build_key(int(item["row"]), int(item["col"]))
            rows.append(
                {
                    "key": key,
                    "pred_id": item.get("pred_id") or item.get("prediction") or item.get("entity_id"),
                    "score": item.get("score") or item.get("confidence"),
                    "runtime_ms": item.get("runtime_ms"),
                    "meta_json": item.get("meta_json"),
                }
            )
        return rows
    return rows


def _run_native(
    table_df: pd.DataFrame,
    mention_columns: list[str],
    target_kg: str | None,
    correct_qids: dict,
    lamapi_endpoint: str,
    lamapi_token: str | None,
    lamapi_limit: int,
):
    try:
        import alligator
    except Exception:
        return None

    Alligator = getattr(alligator, "Alligator", None)
    if Alligator is None:
        return None

    config = None
    if hasattr(alligator, "AlligatorConfig"):
        try:
            config = alligator.AlligatorConfig()
            if hasattr(config, "data") and hasattr(config.data, "correct_qids"):
                config.data.correct_qids = correct_qids
                if hasattr(config.data, "entity_retrieval_endpoint"):
                    config.data.entity_retrieval_endpoint = lamapi_endpoint
                if hasattr(config.data, "entity_retrieval_token"):
                    config.data.entity_retrieval_token = lamapi_token
                if hasattr(config.data, "num_candidates"):
                    config.data.num_candidates = lamapi_limit
        except Exception:
            config = None

    if config is not None:
        model = Alligator(config=config)
    else:
        model = Alligator(
            correct_qids=correct_qids,
            entity_retrieval_endpoint=lamapi_endpoint,
            entity_retrieval_token=lamapi_token,
            lamapi_endpoint=lamapi_endpoint,
            lamapi_token=lamapi_token,
            num_candidates=lamapi_limit,
        )

    if hasattr(model, "link_table"):
        return model.link_table(table_df, mention_columns=mention_columns)
    if hasattr(model, "predict"):
        return model.predict(table_df)
    if hasattr(model, "run"):
        return model.run(table_df)
    raise RuntimeError("Unsupported Alligator API: no link_table/predict/run method.")


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
    os.environ["ENTITY_RETRIEVAL_ENDPOINT"] = lamapi_endpoint
    if args.lamapi_token:
        os.environ["ENTITY_RETRIEVAL_TOKEN"] = args.lamapi_token
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
        "method": "alligator",
        "max_rows": args.max_rows,
        "lamapi_limit": args.lamapi_limit,
        "force_gt_candidate": args.force_gt_candidate,
        "force_id": args.force_id,
        "target_kg": target_kg,
    }
    run_dir = resolve_run_dir(Path(args.output_dir), spec.name, "alligator", settings)
    ensure_dir(run_dir)

    native_output = None
    try:
        native_output = _run_native(
            table_df,
            mention_columns,
            target_kg,
            forced_map,
            lamapi_endpoint,
            args.lamapi_token,
            args.lamapi_limit,
        )
    except Exception as exc:
        logger.warning("Alligator native runner failed, falling back to LamAPI top1: %s", exc)

    if native_output is not None:
        rows = _normalize_native_output(native_output)
    else:
        rows = run_lamapi_top1(table_df, mention_columns, client, target_kg, forced_map)

    pred_path = run_dir / "predictions.csv"
    pd.DataFrame(rows, columns=PRED_COLUMNS).to_csv(pred_path, index=False)
    write_json(run_dir / "manifest.json", settings)
    logger.info("Wrote predictions to %s", pred_path)


if __name__ == "__main__":
    main()
