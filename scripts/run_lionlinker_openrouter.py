#!/usr/bin/env python3
"""Run LionLinker with OpenRouter on a dataset slice and write predictions."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from lion_linker.lion_linker import LionLinker
from lion_linker.retrievers import LamapiClient, RetrieverClient


DATASETS = {
    "movies": {
        "default_input": "data/Movie/movies_test.csv",
        "default_gt": "data/Movie/el_movies_gt_wikidata.csv",
        "mention_col": "title",
        "table_name": "movies_test",
        "mapping": "data/mention_to_qid/movies_mention_to_qids.json",
        "kg_name": "wikidata",
    },
    "companies_wikidata": {
        "default_input": "data/Company/companies_test.csv",
        "default_gt": "data/Company/el_companies_gt_wikidata.csv",
        "mention_col": "company",
        "table_name": "companies_test",
        "mapping": "data/mention_to_qid/companies_mention_to_qids.json",
        "kg_name": "wikidata",
    },
    "companies_crunchbase": {
        "default_input": "data/Company/companies_test.csv",
        "default_gt": "data/Company/el_companies_gt_crunchbase.csv",
        "mention_col": "company",
        "table_name": "companies_test",
        "mapping": "data/mention_to_qid/companies_mention_to_qids_crunchbase.json",
        "kg_name": "crunchbase",
    },
    "sn": {
        "default_input": "data/SN/SN_test.csv",
        "default_gt": "data/SN/SN_gt.csv",
        "mention_col": "buyer",
        "table_name": "SN_test",
        "mapping": "data/mention_to_qid/SN_mention_to_qids.json",
        "kg_name": "wikidata",
    },
}

PROMPT_SCHEMA_MAP = {
    "base": "base",
    "structured": "detailed",
    "fewshot": "few_shot",
}


@dataclass
class DatasetConfig:
    input_csv: Path
    gt_csv: Optional[Path]
    mention_col: str
    table_name: str
    mapping_path: Optional[Path]
    kg_name: str


class MappingRetriever(RetrieverClient):
    def __init__(self, mention_to_ids: Dict[str, List[str]], num_candidates: int = 10):
        super().__init__(endpoint="local", num_candidates=num_candidates)
        self.mention_to_ids = mention_to_ids

    def _build_candidates(self, ids: List[str]) -> List[Dict]:
        output = []
        for candidate_id in ids[: self.num_candidates]:
            output.append(
                {
                    "id": candidate_id,
                    "name": candidate_id,
                    "description": None,
                    "types": [],
                }
            )
        return output

    async def fetch_entities(self, mention: str, session, **kwargs):
        forced_ids = kwargs.get("forced_ids")
        if forced_ids is None:
            ids = self.mention_to_ids.get(mention, [])
        else:
            ids = forced_ids if isinstance(forced_ids, list) else [forced_ids]
        return self._build_candidates(ids)

    async def fetch_multiple_entities(self, mentions: List[str], **kwargs):
        mention_to_qids = kwargs.get("mention_to_qids", {})
        output = {}
        for mention in mentions:
            forced_ids = mention_to_qids.get(mention)
            if forced_ids is None:
                ids = self.mention_to_ids.get(mention, [])
            else:
                ids = forced_ids if isinstance(forced_ids, list) else [forced_ids]
            output[mention] = self._build_candidates(ids)
        return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LionLinker with OpenRouter on a dataset slice."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(DATASETS.keys()),
        help="Dataset to run.",
    )
    parser.add_argument("--input_csv", type=str, default=None, help="Override input CSV path.")
    parser.add_argument("--gt_csv", type=str, default=None, help="Optional GT CSV path.")
    parser.add_argument(
        "--mention_col",
        type=str,
        default=None,
        help="Column containing mentions (default depends on dataset).",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="Number of rows to process (0 means all rows).",
    )
    parser.add_argument("--model", type=str, required=True, help="OpenRouter model id.")
    parser.add_argument(
        "--prompt_schema",
        type=str,
        choices=sorted(PROMPT_SCHEMA_MAP.keys()),
        default="base",
    )
    parser.add_argument("--candidates_limit", type=int, default=10)
    parser.add_argument("--max_concurrency", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--openrouter_api_key",
        type=str,
        default=None,
        help="Optional OpenRouter API key (otherwise uses OPENROUTER_API_KEY).",
    )
    return parser.parse_args()


def resolve_dataset_config(args: argparse.Namespace) -> DatasetConfig:
    config = DATASETS[args.dataset]
    input_csv = Path(args.input_csv or config["default_input"])
    gt_csv = Path(args.gt_csv or config["default_gt"]) if config.get("default_gt") else None
    mention_col = args.mention_col or config["mention_col"]
    mapping_path = Path(config["mapping"]) if config.get("mapping") else None
    table_name = config["table_name"]
    if args.input_csv:
        table_name = input_csv.stem
    return DatasetConfig(
        input_csv=input_csv,
        gt_csv=gt_csv,
        mention_col=mention_col,
        table_name=table_name,
        mapping_path=mapping_path if mapping_path and mapping_path.exists() else None,
        kg_name=config["kg_name"],
    )


def get_openrouter_key(args: argparse.Namespace) -> str:
    if args.openrouter_api_key:
        return args.openrouter_api_key
    env_key = os.getenv("OPENROUTER_API_KEY")
    if env_key:
        return env_key
    raise RuntimeError("Missing OPENROUTER_API_KEY. Set it or pass --openrouter_api_key.")


def load_mapping(path: Path) -> Dict[str, List[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def git_commit_hash() -> Optional[str]:
    git_dir = Path(".git")
    if not git_dir.exists():
        return None
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def resolve_prediction_column(df: pd.DataFrame, mention_col: str) -> str:
    expected = f"{mention_col}_pred_id"
    if expected in df.columns:
        return expected
    pred_cols = [col for col in df.columns if col.endswith("_pred_id")]
    if len(pred_cols) == 1:
        return pred_cols[0]
    raise ValueError(
        "Prediction column not found. Expected column "
        f"'{expected}' or a single '*_pred_id' column."
    )


def write_manifest(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.n_rows < 0:
        raise ValueError("--n_rows must be >= 0")

    config = resolve_dataset_config(args)
    if not config.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {config.input_csv}")

    api_key = get_openrouter_key(args)

    prompt_template = PROMPT_SCHEMA_MAP[args.prompt_schema]
    run_id = args.run_id or f"{args.dataset}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir)
    run_dir = output_dir / run_id
    ensure_dir(run_dir)

    print(f"[LionLinker] Dataset: {args.dataset}")
    print(f"[LionLinker] Input CSV: {config.input_csv}")
    print(f"[LionLinker] Output dir: {run_dir}")

    df = pd.read_csv(config.input_csv)
    mention_col = config.mention_col
    if mention_col not in df.columns:
        matches = [col for col in df.columns if col.lower() == mention_col.lower()]
        if len(matches) == 1:
            mention_col = matches[0]
        else:
            raise ValueError(
                f"Column '{mention_col}' not found in input CSV. "
                f"Available columns: {', '.join(df.columns)}"
            )

    if args.n_rows > 0:
        df = df.head(args.n_rows)
        print(f"[LionLinker] Using first {args.n_rows} rows")
    else:
        print(f"[LionLinker] Using all {len(df)} rows")

    slice_path = run_dir / "input_slice.csv"
    df.to_csv(slice_path, index=False)

    if config.mapping_path and config.mapping_path.exists():
        mention_to_ids = load_mapping(config.mapping_path)
        retriever: RetrieverClient = MappingRetriever(
            mention_to_ids, num_candidates=args.candidates_limit
        )
        retriever_name = "mapping"
        print(f"[LionLinker] Retriever: mapping ({config.mapping_path})")
    else:
        endpoint = os.getenv("ENTITY_RETRIEVAL_ENDPOINT")
        token = os.getenv("ENTITY_RETRIEVAL_TOKEN")
        if not endpoint:
            raise RuntimeError(
                "Missing ENTITY_RETRIEVAL_ENDPOINT for LamAPI retrieval. "
                "Set it in the environment."
            )
        retriever = LamapiClient(
            endpoint=endpoint,
            token=token,
            num_candidates=args.candidates_limit,
        )
        retriever_name = "lamapi"
        print("[LionLinker] Retriever: LamAPI client (online)")

    output_csv = run_dir / "predictions.csv"
    linker = LionLinker(
        input_csv=str(slice_path),
        model_name=args.model,
        retriever=retriever,
        output_csv=str(output_csv),
        mention_columns=[mention_col],
        model_api_provider="openrouter",
        model_api_key=api_key,
        prompt_template=prompt_template,
        kg_name=config.kg_name,
    )

    asyncio.run(linker.run())

    print("[LionLinker] Post-processing predictions")
    raw_df = pd.read_csv(output_csv)
    pred_col = resolve_prediction_column(raw_df, mention_col)

    if "id_row" in raw_df.columns:
        id_row = raw_df["id_row"]
    else:
        id_row = raw_df.index

    id_col = df.columns.get_loc(mention_col)
    table_name = config.table_name
    output_df = pd.DataFrame(
        {
            "id": [f"{table_name}-{row}-{id_col}" for row in id_row],
            "tableName": table_name,
            "idRow": id_row,
            "idCol": id_col,
            "mention": raw_df[mention_col],
            "predicted_id": raw_df[pred_col],
        }
    )
    output_df.to_csv(output_csv, index=False)

    versions = {"python": sys.version.split()[0]}
    try:
        import lion_linker

        versions["lion_linker"] = getattr(lion_linker, "__version__", "unknown")
    except Exception:
        versions["lion_linker"] = "unknown"

    manifest = {
        "dataset": args.dataset,
        "model": args.model,
        "prompt_schema": args.prompt_schema,
        "prompt_template": prompt_template,
        "candidates_limit": args.candidates_limit,
        "max_concurrency": args.max_concurrency,
        "n_rows": args.n_rows,
        "input_csv": str(config.input_csv),
        "sliced_csv": str(slice_path),
        "output_csv": str(output_csv),
        "gt_csv": str(config.gt_csv) if config.gt_csv else None,
        "mention_col": mention_col,
        "table_name": table_name,
        "retriever": retriever_name,
        "seed": args.seed,
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_commit": git_commit_hash(),
        "versions": versions,
    }
    write_manifest(run_dir / "manifest.json", manifest)
    print(f"[LionLinker] Done. Predictions at {output_csv}")


if __name__ == "__main__":
    main()
