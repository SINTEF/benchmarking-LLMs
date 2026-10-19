"""CLI for background runs.

    python -m experiments.runs worker                       # execute queued runs (keep it running)
    python -m experiments.runs submit --dataset mv --model openai/gpt-oss-120b --concurrency 32
    python -m experiments.runs list
    python -m experiments.runs cancel <run_id>
    python -m experiments.runs resume <run_id>
"""

from __future__ import annotations

import argparse
import time

from experiments.common.env import load_dotenv
from experiments.cost.tasks import REPO_ROOT
from experiments.runs.config import SCHEMAS, RunConfig
from experiments.runs.eta import fmt_duration, live_eta
from experiments.runs.store import RunStore
from experiments.runs.worker import run_worker


def add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dataset", required=True, help="spec name (mv) or source key (semtab:HardTablesR2)")
    p.add_argument("--model", required=True, help="OpenRouter model id")
    p.add_argument("--schema", choices=SCHEMAS, default="base")
    p.add_argument("--provider", default=None, help="pin an OpenRouter endpoint tag, e.g. groq")
    p.add_argument("--provider-policy", choices=["byok", "byok_first", "routing"], default="byok",
                   help="byok (default): only BYOK providers; byok_first: then shared capacity; routing")
    p.add_argument("--limit", type=int, default=0, help="first N rows (spec) / cells (SemTab)")
    p.add_argument("--lamapi-limit", type=int, default=10)
    p.add_argument("--no-force-gt", action="store_true", help="do NOT force the GT ids into the candidates")
    p.add_argument("--force-id", action="append", default=[], help="extra id forced into every candidate list")
    p.add_argument("--concurrency", type=int, default=16, help="parallel LLM requests")
    p.add_argument("--max-rpm", type=int, default=0, help="requests/minute cap (0 = none)")
    p.add_argument("--lamapi-concurrency", type=int, default=8)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--datasets-root", default=None)


def config_from_args(a: argparse.Namespace) -> RunConfig:
    return RunConfig(
        dataset=a.dataset, model=a.model, schema=a.schema, provider=a.provider,
        provider_policy="pin" if a.provider else a.provider_policy, limit=a.limit,
        lamapi_limit=a.lamapi_limit, force_gt=not a.no_force_gt, force_ids=a.force_id,
        max_concurrency=a.concurrency, max_rpm=a.max_rpm, lamapi_concurrency=a.lamapi_concurrency,
        output_dir=a.output_dir, datasets_root=a.datasets_root,
    )


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worker", help="execute queued runs")
    w.add_argument("--once", action="store_true", help="run at most one queued run, then exit")
    s = sub.add_parser("submit", help="queue a run")
    add_run_args(s)
    sub.add_parser("list", help="show runs")
    c = sub.add_parser("cancel")
    c.add_argument("run_id")
    r = sub.add_parser("resume", help="queue a finished/cancelled/failed run again (continues where it stopped)")
    r.add_argument("run_id")
    a = parser.parse_args()

    store = RunStore()
    if a.cmd == "worker":
        run_worker(store, once=a.once)
    elif a.cmd == "submit":
        cfg = config_from_args(a)
        cfg.resolve_source()  # fail fast on a bad dataset name
        print(store.submit(cfg.to_dict(), label=cfg.label()))
    elif a.cmd == "list":
        store.mark_stale()
        for run in store.list(50):
            p = run.progress
            eta = live_eta(p, store.ticks(run.id, since=time.time() - 120)) if run.status == "running" else {}
            print(
                f"{run.id}  {run.status:<11} {p.get('n_done', 0):>7}/{p.get('n_tasks', '?'):<7} "
                f"${p.get('cost_usd', 0):>8.4f}  eta {fmt_duration(eta.get('eta_s')):>8}  {run.label}"
            )
    elif a.cmd == "cancel":
        store.cancel(a.run_id)
    elif a.cmd == "resume":
        store.requeue(a.run_id)


if __name__ == "__main__":
    main()
