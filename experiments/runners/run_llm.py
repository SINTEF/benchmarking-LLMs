#!/usr/bin/env python3
"""Run the LLM linker (LionLinker prompts + LamAPI candidates) on one dataset, in this process.

Requests go out in parallel (``--max-concurrency``, optional ``--max-rpm``) and the GT ids of every
cell are forced into its candidate list unless ``--no-force-gt`` is given. The run is registered
in the run store, so the console's Runs page monitors it like a queued run.

To run in the background instead: ``python -m experiments.runs submit ...`` + ``make worker``.
"""

from __future__ import annotations

import argparse
import os
import socket
from dataclasses import replace
from pathlib import Path

from experiments.common.env import load_dotenv
from experiments.common.logging import get_logger
from experiments.cost.tasks import REPO_ROOT
from experiments.runners.common import add_common_args
from experiments.runs.config import SCHEMAS, RunConfig
from experiments.runs.executor import RunExecutor
from experiments.runs.store import RunStore

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Any dataset the console lists: repo specs (mv) or SemTab folders under DATASETS_ROOT (semtab:2T_2020).
    parser.add_argument("--dataset", required=True, help="repo dataset (mv, sn...) or semtab:<folder>")
    parser.add_argument("--table-path", type=str, default=None)
    parser.add_argument("--gt-path", type=str, default=None)
    parser.add_argument("--target-kg", type=str, default=None)
    parser.add_argument("--datasets-root", default=None, help="SemTab datasets folder (default: DATASETS_ROOT)")
    add_common_args(parser)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--provider", default="openrouter", choices=["openrouter"], help="LLM API (OpenRouter only)")
    parser.add_argument("--pin-provider", default=None, help="OpenRouter endpoint tag to pin, e.g. groq")
    parser.add_argument(
        "--provider-policy", choices=["byok", "byok_first", "routing"], default="byok",
        help="byok: only your BYOK providers, back off on 429 (default); byok_first: then shared capacity; "
        "routing: let OpenRouter decide. See experiments/runs/byok.py",
    )
    parser.add_argument("--prompt-schema", choices=SCHEMAS, default="base")
    parser.add_argument("--model-api-key", default=None, type=str)
    parser.add_argument("--max-concurrency", type=int, default=16, help="parallel LLM (OpenRouter) requests")
    parser.add_argument("--max-rpm", type=int, default=0, help="LLM requests per minute cap (0 = none)")
    parser.add_argument("--lamapi-concurrency", type=int, default=8, help="parallel LamAPI lookups")
    parser.add_argument(
        "--no-force-gt", action="store_true",
        help="do not force GT ids into the candidates (forced by default; --force-gt-candidate is implied)",
    )
    parser.add_argument("--no-store", action="store_true", help="do not register the run in the run store")
    return parser.parse_args()


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    args = parse_args()
    if args.max_rows < 0:
        raise ValueError("--max-rows must be >= 0")
    if args.lamapi_endpoint:
        os.environ["ENTITY_RETRIEVAL_ENDPOINT"] = args.lamapi_endpoint
    if args.lamapi_token:
        os.environ["ENTITY_RETRIEVAL_TOKEN"] = args.lamapi_token
    if args.lamapi_cache_dir:
        os.environ["LAMAPI_CACHE_DIR"] = args.lamapi_cache_dir

    config = RunConfig(
        dataset=args.dataset,
        model=args.model,
        schema=args.prompt_schema,
        provider=args.pin_provider,
        provider_policy="pin" if args.pin_provider else args.provider_policy,
        limit=args.max_rows,
        lamapi_limit=args.lamapi_limit,
        force_gt=not args.no_force_gt,
        force_ids=list(args.force_id),
        max_concurrency=args.max_concurrency,
        max_rpm=args.max_rpm,
        lamapi_concurrency=args.lamapi_concurrency,
        output_dir=args.output_dir,
        datasets_root=args.datasets_root,
    )
    source = config.resolve_source()
    overrides = {}
    if args.table_path:
        overrides["table_path"] = Path(args.table_path)
    if args.gt_path:
        overrides["gt_path"] = Path(args.gt_path)
    if args.target_kg:
        overrides["target_kg"] = args.target_kg
    if overrides:
        source = replace(source, **overrides)

    store = run_id = None
    if not args.no_store:
        store = RunStore()
        run_id = store.register_running(config.to_dict(), config.label(), f"cli-{socket.gethostname()}-{os.getpid()}")
    executor = RunExecutor(config, store, run_id, api_key=args.model_api_key, source=source)
    try:
        manifest = executor.run()
    except KeyboardInterrupt:
        if store and run_id:
            store.finish(run_id, "interrupted", "stopped from the command line; resume to continue")
        raise
    report = manifest.get("report") or {}
    logger.info(
        "%s: %s · P %s R %s F1 %s · $%.4f -> %s", config.label(), manifest["status"],
        *(f"{report[k]:.4f}" if k in report else "—" for k in ("precision", "recall", "f1")),
        manifest["progress"]["cost_usd"], executor.run_dir / "predictions.csv",
    )
    if manifest["status"] != "done":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
