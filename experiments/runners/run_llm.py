#!/usr/bin/env python3
"""Run LionLinker with LamAPI retrieval."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.common.env import load_dotenv
from experiments.common.io import ensure_dir, write_json
from experiments.common.logging import get_logger
from experiments.common.paths import resolve_run_dir
from experiments.common.schemas import PRED_COLUMNS
from experiments.forcing import build_mention_to_qids_from_table
from experiments.retrieval.lamapi_client import LamapiClient
from experiments.runners.common import (
    add_common_args,
    add_dataset_args,
    resolve_mention_columns,
    resolve_paths,
    resolve_spec,
    resolve_target_kg,
)

logger = get_logger(__name__)

PROMPT_SCHEMA_MAP = {
    "base": "base",
    "structured": "detailed",
    "fewshot": "few_shot",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LionLinker with LamAPI retrieval.")
    add_dataset_args(parser)
    add_common_args(parser)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--provider", default="openrouter", type=str)
    parser.add_argument("--prompt-schema", choices=sorted(PROMPT_SCHEMA_MAP.keys()), default="base")
    parser.add_argument("--model-api-key", default=None, type=str)
    parser.add_argument("--max-concurrency", type=int, default=32)
    return parser.parse_args()


def resolve_prediction_columns(df: pd.DataFrame, mention_columns: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    pred_cols = [col for col in df.columns if col.endswith("_pred_id")]
    for mention_col in mention_columns:
        expected = f"{mention_col}_pred_id"
        if expected in df.columns:
            mapping[mention_col] = expected
            continue
        matches = [col for col in pred_cols if col.lower() == expected.lower()]
        if len(matches) == 1:
            mapping[mention_col] = matches[0]
            continue
    if not mapping and len(pred_cols) == 1 and len(mention_columns) == 1:
        mapping[mention_columns[0]] = pred_cols[0]
    if not mapping:
        raise ValueError("Could not resolve prediction columns in LionLinker output.")
    return mapping


def resolve_lionlinker_normalizer():
    try:
        from lion_linker.utils import normalize_mention

        return normalize_mention
    except Exception:
        return None


class LionLinkerLamapiRetriever:
    def __init__(self, client: LamapiClient, target_kg: str | None):
        self.client = client
        self.target_kg = target_kg

    async def fetch_entities(self, mention: str, session=None, **kwargs):
        forced_ids = kwargs.get("forced_ids")
        candidates = self.client.fetch_entities(mention, ids=forced_ids, target_kg=self.target_kg)
        return [_to_lionlinker_candidate(candidate) for candidate in candidates]

    async def fetch_multiple_entities(self, mentions: list[str], **kwargs):
        mention_to_qids = kwargs.get("mention_to_qids", {})
        output: dict[str, list[dict[str, Any]]] = {}
        for mention in mentions:
            forced_ids = mention_to_qids.get(mention)
            candidates = self.client.fetch_entities(mention, ids=forced_ids, target_kg=self.target_kg)
            output[mention] = [_to_lionlinker_candidate(candidate) for candidate in candidates]
        return output


def _to_lionlinker_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": candidate.get("id"),
        "name": candidate.get("label") or candidate.get("id"),
        "description": candidate.get("description"),
        "types": candidate.get("types") or [],
    }


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.max_rows < 0:
        raise ValueError("--max-rows must be >= 0")

    spec = resolve_spec(args)
    table_path, gt_path = resolve_paths(args, spec)
    target_kg = resolve_target_kg(args, spec)

    try:
        from lion_linker.lion_linker import LionLinker
    except Exception as exc:
        raise RuntimeError("lion_linker is required for run_llm.py") from exc

    table_df = pd.read_csv(table_path)
    mention_columns = resolve_mention_columns(table_df, spec.mention_columns)
    if args.max_rows:
        table_df = table_df.head(args.max_rows)

    normalizer = resolve_lionlinker_normalizer()
    mention_to_qids = None
    if args.force_gt_candidate:
        if not gt_path:
            raise ValueError("--force-gt-candidate requires --gt-path or dataset GT.")
        mention_to_qids = build_mention_to_qids_from_table(
            table_path=table_path,
            gt_path=gt_path,
            mention_columns=mention_columns,
            normalizer=normalizer,
        )
        mentions = set()
        for col in mention_columns:
            for value in table_df[col].dropna().astype(str).tolist():
                mentions.add(normalizer(value) if normalizer else value)
        mention_to_qids = {
            mention: value for mention, value in mention_to_qids.items() if mention in mentions
        }

    if args.force_id:
        mentions = []
        for col in mention_columns:
            mentions.extend(table_df[col].dropna().astype(str).tolist())
        if mention_to_qids is None:
            mention_to_qids = {}
        for mention in mentions:
            mention_key = normalizer(mention) if normalizer else mention
            existing = mention_to_qids.get(mention_key)
            merged = []
            if existing:
                merged.extend(existing if isinstance(existing, list) else [existing])
            for force_id in args.force_id:
                if force_id not in merged:
                    merged.append(force_id)
            mention_to_qids[mention_key] = merged

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
        "method": "llm",
        "model": args.model,
        "provider": args.provider,
        "prompt_schema": args.prompt_schema,
        "max_concurrency": args.max_concurrency,
        "max_rows": args.max_rows,
        "lamapi_limit": args.lamapi_limit,
        "force_gt_candidate": args.force_gt_candidate,
        "force_id": args.force_id,
        "target_kg": target_kg,
    }
    run_dir = resolve_run_dir(Path(args.output_dir), spec.name, "llm", settings)
    ensure_dir(run_dir)

    slice_path = run_dir / "input_slice.csv"
    table_df.to_csv(slice_path, index=False)

    retriever = LionLinkerLamapiRetriever(client, target_kg)
    output_raw = run_dir / "lionlinker_raw.csv"
    prompt_template = PROMPT_SCHEMA_MAP[args.prompt_schema]
    model_api_key = args.model_api_key or os.getenv("OPENROUTER_API_KEY")
    if not model_api_key:
        raise RuntimeError("Missing model API key. Pass --model-api-key or set OPENROUTER_API_KEY.")

    linker = LionLinker(
        input_csv=str(slice_path),
        model_name=args.model,
        retriever=retriever,
        output_csv=str(output_raw),
        mention_columns=mention_columns,
        model_api_provider=args.provider,
        model_api_key=model_api_key,
        prompt_template=prompt_template,
        kg_name=target_kg or "generic",
        mention_to_qids=mention_to_qids or {},
    )
    asyncio.run(linker.run())

    raw_df = pd.read_csv(output_raw)
    pred_cols = resolve_prediction_columns(raw_df, mention_columns)

    rows = []
    for row_idx in range(len(raw_df)):
        if "id_row" in raw_df.columns:
            base_row = int(raw_df.loc[row_idx, "id_row"])
        elif "idRow" in raw_df.columns:
            base_row = int(raw_df.loc[row_idx, "idRow"])
        else:
            base_row = row_idx
        for mention_col in mention_columns:
            col_idx = table_df.columns.get_loc(mention_col)
            key = f"{base_row}-{col_idx}"
            pred_col = pred_cols[mention_col]
            pred_id = raw_df.loc[row_idx, pred_col]
            rows.append(
                {
                    "key": key,
                    "pred_id": pred_id,
                    "score": None,
                    "runtime_ms": None,
                    "meta_json": json.dumps(
                        {
                            "mention": raw_df.loc[row_idx, mention_col],
                            "model": args.model,
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
