"""Shared CLI for the classic table linkers (Crocodile, Alligator), executed in this process.

They need MongoDB and tensorflow==2.18 (Python <= 3.12), i.e. the ``linkers`` docker-compose image:

    docker compose run --rm linkers python -m experiments.runners.run_crocodile --dataset mv --max-rows 50

There is no silent fallback any more: if the library is missing the run fails instead of
reporting LamAPI's top candidate under the linker's name.
"""

from __future__ import annotations

import argparse
import os
import socket

from experiments.common.env import load_dotenv
from experiments.common.logging import get_logger
from experiments.cost.tasks import REPO_ROOT
from experiments.runners.common import add_common_args
from experiments.runs.config import METHODS, RunConfig
from experiments.runs.store import RunStore

logger = get_logger(__name__)


def main(method: str) -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=f"Run {METHODS[method]} on a dataset (GT ids forced by default).")
    parser.add_argument("--dataset", required=True, help="repo dataset (mv, sn...) or semtab:<folder>")
    parser.add_argument("--datasets-root", default=None)
    parser.add_argument("--workers", type=int, default=4, help="worker processes per table")
    parser.add_argument("--no-force-gt", action="store_true", help="do not force GT ids into the candidates")
    parser.add_argument("--no-store", action="store_true", help="do not register the run in the run store")
    add_common_args(parser)
    args = parser.parse_args()
    try:
        __import__(method)
    except ImportError as exc:
        raise SystemExit(
            f"{METHODS[method]} is not installed here ({exc}). Run it in the linkers image: "
            f"docker compose run --rm linkers python -m experiments.runners.run_{method} --dataset ..."
        ) from exc
    if args.lamapi_endpoint:
        os.environ["ENTITY_RETRIEVAL_ENDPOINT"] = args.lamapi_endpoint
    if args.lamapi_token:
        os.environ["ENTITY_RETRIEVAL_TOKEN"] = args.lamapi_token

    from experiments.runs.linkers import LinkerExecutor

    config = RunConfig(
        dataset=args.dataset, method=method, limit=args.max_rows, lamapi_limit=args.lamapi_limit,
        force_gt=not args.no_force_gt, force_ids=list(args.force_id), linker_workers=args.workers,
        output_dir=args.output_dir, datasets_root=args.datasets_root,
    )
    store = run_id = None
    if not args.no_store:
        store = RunStore()
        run_id = store.register_running(config.to_dict(), config.label(), f"cli-{socket.gethostname()}-{os.getpid()}")
    executor = LinkerExecutor(config, store, run_id)
    try:
        manifest = executor.run()
    except KeyboardInterrupt:
        if store and run_id:
            store.finish(run_id, "interrupted", "stopped from the command line; resume to continue")
        raise
    report = manifest.get("report") or {}
    logger.info(
        "%s: %s · P %s R %s F1 %s -> %s", config.label(), manifest["status"],
        *(f"{report[k]:.4f}" if k in report else "—" for k in ("precision", "recall", "f1")),
        executor.run_dir / "predictions.csv",
    )
    if manifest["status"] != "done":
        raise SystemExit(1)
