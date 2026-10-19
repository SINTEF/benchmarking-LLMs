"""Runs page: queue LLM runs with a time estimate, then monitor them live."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from experiments.cost.candidates import make_lamapi_clients
from experiments.cost.pricing import load_endpoints, load_prices
from experiments.cost.tasks import discover_sources, load_taskset
from experiments.runs.config import RunConfig
from experiments.runs.eta import estimate_duration, fmt_duration, live_eta, lookup_plan, seconds_per_call_for
from experiments.runs.store import ACTIVE, CANCELLING, DONE, FINISHED, QUEUED, RUNNING, RunStore

SCHEMAS = {"base": "Base", "structured": "Structured (detailed)", "fewshot": "Few-shot"}
DEFAULT_MODEL = "openai/gpt-oss-120b"
ROUTING, CHEAPEST = "OpenRouter routing", "Cheapest healthy provider (pinned)"
STATUS_ICON = {
    "queued": ":material/schedule:", "running": ":material/play_circle:", "cancelling": ":material/hourglass_top:",
    "done": ":material/check_circle:", "failed": ":material/error:", "cancelled": ":material/cancel:",
    "interrupted": ":material/pause_circle:",
}


@st.cache_resource
def get_store() -> RunStore:
    return RunStore()


@st.cache_data(ttl=300, show_spinner=False)
def get_sources(root: str):
    return discover_sources(Path(root).expanduser() if root else None)


@st.cache_data(ttl=3600, show_spinner=False)
def get_prices():
    return load_prices()


@st.cache_data(ttl=3600, show_spinner=False)
def get_endpoints(model_id: str):
    return load_endpoints(model_id)


@st.cache_data(ttl=60, show_spinner="Checking which LamAPI lookups are cached...")
def get_plan(source_key: str, root: str, limit: int, lamapi_limit: int, force_gt: bool, force_ids: tuple):
    source = next(s for s in get_sources(root) if s.key == source_key)
    ts = load_taskset(source, limit)
    offline, _ = make_lamapi_clients(lamapi_limit)
    plan = lookup_plan(ts, offline, force_gt, force_ids)
    plan["tables"] = ts.n_tables
    return plan


def usd(x: float | None) -> str:
    if x is None:
        return "—"
    return f"${x:,.2f}" if x >= 1 else f"${x:.4f}"


def read_report(run) -> dict | None:
    if not run.run_dir:
        return None
    path = Path(run.run_dir) / "report.json"
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        return None


store = get_store()
st.title("Runs")

# --------------------------------------------------------------------------- launch

all_runs = store.list(300)
with st.expander("Queue new runs", icon=":material/add_circle:", expanded=not all_runs):
    datasets_root = os.getenv("DATASETS_ROOT", "")
    sources = get_sources(datasets_root)
    by_key = {s.key: s for s in sources}
    prices = get_prices()

    c1, c2 = st.columns([3, 2])
    dataset_keys = c1.multiselect(
        "Datasets", list(by_key), default=[k for k in ["spec:mv"] if k in by_key],
        format_func=lambda k: f"{by_key[k].name}  ·  {'repo' if by_key[k].kind == 'spec' else 'SemTab'}",
    )
    schemas = c2.multiselect("Prompt schemas", list(SCHEMAS), default=["base"], format_func=SCHEMAS.get)
    models = st.multiselect(
        "Models (OpenRouter)", sorted(prices.models) or [DEFAULT_MODEL],
        default=[DEFAULT_MODEL] if DEFAULT_MODEL in prices.models or not prices.models else [],
        placeholder="Type to search, e.g. gpt-oss, llama, qwen...",
    )
    provider_mode = st.radio(
        "Provider", [ROUTING, CHEAPEST], horizontal=True,
        help="Routing lets OpenRouter pick (and fall back between) providers. Pinning the cheapest healthy "
        "provider makes the price predictable but a single provider's rate limits apply.",
    )

    c1, c2, c3, c4 = st.columns(4)
    limit = c1.number_input("Limit (first N rows / cells, 0 = all)", min_value=0, value=0, step=100)
    lamapi_limit = c2.number_input("Candidates per mention", min_value=1, max_value=100, value=10)
    concurrency = c3.number_input(
        "Parallel requests", min_value=1, max_value=512, value=16,
        help="Requests in flight at once. OpenRouter has no fixed limit on paid models, but providers answer "
        "429 when pushed too hard; the run then pauses all requests and retries.",
    )
    max_rpm = c4.number_input("Max requests / min (0 = no cap)", min_value=0, value=0, step=60)

    c1, c2 = st.columns([1, 2])
    force_gt = c1.checkbox(
        "Force GT candidates", value=True,
        help="Every GT id of a cell is passed to LamAPI (`ids`, space-separated), so the correct entity is "
        "always among the candidates. The run reports how many cells still miss it.",
    )
    extra_ids = c2.text_input("Extra ids forced into every candidate list", placeholder="Q42 Q64")
    force_ids = tuple(extra_ids.replace(",", " ").split())

    combos = [(d, s, m) for d in dataset_keys for s in schemas for m in models]

    # ---- time estimate
    if combos:
        rows = []
        for d, s, m in combos:
            plan = get_plan(d, datasets_root, int(limit), int(lamapi_limit), force_gt, force_ids)
            price = prices.models.get(m)
            pinned = None
            latency = throughput = None
            if provider_mode == CHEAPEST:
                eps = get_endpoints(m)
                best = eps.cheapest(700, 400 if price and price.reasoning else 10)
                if best:
                    pinned, latency, throughput = best.tag, best.latency_s, best.throughput_tps
            spc, spc_src = seconds_per_call_for(
                store, m, pinned, bool(price and price.reasoning), latency_s=latency, throughput_tps=throughput,
                output_tokens=400 if price and price.reasoning else 10,
            )
            calls = plan["cells"] + plan["tables"]
            est = estimate_duration(
                calls, spc, int(concurrency), int(max_rpm), lookups_to_fetch=plan["to_fetch"],
                seconds_per_lookup=0.6, seconds_per_call_source=spc_src,
            )
            rows.append({
                "dataset": by_key[d].name, "schema": s, "model": m, "provider": pinned or "routing",
                "calls": calls, "uncached lookups": plan["to_fetch"], "s/call": round(spc, 2),
                "latency source": spc_src, "bottleneck": est.bottleneck, "duration": fmt_duration(est.seconds),
                "_seconds": est.seconds, "_cfg": RunConfig(
                    dataset=d, model=m, schema=s, provider=pinned, limit=int(limit), lamapi_limit=int(lamapi_limit),
                    force_gt=force_gt, force_ids=list(force_ids), max_concurrency=int(concurrency),
                    max_rpm=int(max_rpm), datasets_root=datasets_root or None,
                ), "_est": est,
            })
        table = pd.DataFrame(rows)
        total_s = table["_seconds"].sum()
        st.markdown(
            f"**{len(rows)} run(s), {int(table['calls'].sum()):,} LLM calls, about {fmt_duration(total_s)}** "
            "if they run one after another on one worker."
        )
        st.dataframe(
            table.drop(columns=["_seconds", "_cfg", "_est"]), hide_index=True, width="stretch",
            column_config={
                "calls": st.column_config.NumberColumn("LLM calls", format="localized",
                                                       help="Non-empty mention cells + one table summary per table."),
                "uncached lookups": st.column_config.NumberColumn(
                    format="localized", help="LamAPI lookups not in data/lamapi_cache yet (fetched during the run)."),
                "s/call": st.column_config.NumberColumn("Seconds / call"),
                "bottleneck": st.column_config.TextColumn(
                    help="parallelism = parallel requests ÷ seconds per call; rpm cap = your requests/min cap; "
                    "LamAPI = fetching uncached candidates is slower than the LLM calls."),
            },
        )
        st.caption(
            "Duration = calls ÷ throughput, where throughput is the lowest of *parallel requests ÷ seconds per call* "
            "and the requests/min cap; uncached LamAPI lookups run alongside. Retries after 429s are not included. "
            "Seconds per call come from calibration or earlier runs of the same model when available. "
            "For cost, use the **Estimate cost** page."
        )
        if not os.getenv("OPENROUTER_API_KEY"):
            st.error("OPENROUTER_API_KEY is not set in .env - runs would fail at once.")
        if st.button(f"Queue {len(rows)} run(s)", type="primary", icon=":material/play_arrow:"):
            for r in rows:
                cfg = r["_cfg"]
                store.submit(cfg.to_dict(), cfg.label(), estimate=r["_est"].to_dict())
            st.toast(f"Queued {len(rows)} run(s).", icon=":material/check:")
            st.rerun()
    else:
        st.info("Pick at least one dataset, prompt schema and model.")


# --------------------------------------------------------------------------- monitor


def active_card(run) -> None:
    p = run.progress
    eta = live_eta(p, store.ticks(run.id, since=time.time() - 180)) if run.status == RUNNING else {}
    n, done = p.get("n_tasks") or 0, p.get("n_done") or 0
    with st.container(border=True):
        top = st.columns([6, 1])
        top[0].markdown(f"{STATUS_ICON.get(run.status, '')} **{run.label}** · `{run.id}` · {run.status}")
        if top[1].button("Cancel", key=f"cancel_{run.id}", icon=":material/stop:", disabled=run.status == CANCELLING):
            store.cancel(run.id)
            st.rerun()
        if run.status == QUEUED:
            est = run.estimate or {}
            st.caption(f"Waiting for a worker · estimated {fmt_duration(est.get('seconds'))} once it starts")
            return
        st.progress(done / n if n else 0.0, text=f"{done:,} / {n:,} cells · {p.get('phase', '')}")
        m = st.columns(6)
        rate = eta.get("rate_per_s")
        m[0].metric("Throughput", f"{rate * 60:,.0f}/min" if rate else "—")
        m[1].metric("ETA", fmt_duration(eta.get("eta_s")), help=f"Elapsed {fmt_duration(run.elapsed_s)}")
        m[2].metric("Cost so far", usd(p.get("cost_usd")), help=f"Projected total: {usd(eta.get('projected_cost_usd'))}")
        m[3].metric(
            "In flight", f"{p.get('in_flight', 0)} / {p.get('max_concurrency', '?')}",
            help="Requests in flight / max parallel requests" + (" · paused after a 429" if p.get("cooling_down") else ""),
        )
        m[4].metric("Errors · 429s", f"{p.get('n_errors', 0)} · {p.get('rate_limited', 0)}",
                    help=f"{p.get('retries', 0)} retries so far")
        gt_cells = p.get("gt_cells") or 0
        m[5].metric("GT in candidates", f"{p.get('gt_present', 0) / gt_cells:.1%}" if gt_cells else "—",
                    help="Cells whose GT id is among the candidates sent to the model.")
        if p.get("cooling_down"):
            st.caption(":material/pause: Provider returned 429 - all requests paused briefly. If this keeps happening, "
                       "lower *parallel requests* or set a requests/min cap.")


def worker_status() -> None:
    store.mark_stale()
    workers = store.workers()
    if workers:
        busy = sum(1 for w in workers if w["current_run"])
        st.caption(f":material/memory: {len(workers)} worker(s) alive, {busy} busy · run store `{store.path}`")
    else:
        st.warning(
            "No worker is running, so queued runs will wait. Start one with `make worker` "
            "(or `docker compose up -d`, which starts it as a service).",
            icon=":material/warning:",
        )


@st.fragment(run_every=3)
def monitor() -> None:
    worker_status()
    runs = store.list(300)
    active = [r for r in runs if r.status in ACTIVE]
    st.subheader(f"Active ({len(active)})")
    if not active:
        st.caption("Nothing running or queued.")
    for run in sorted(active, key=lambda r: (r.status != RUNNING, r.created_at)):
        active_card(run)

    st.subheader("History")
    if not runs:
        st.caption("No runs yet.")
        return
    rows = []
    for r in runs:
        p = r.progress
        report = read_report(r) if r.status in FINISHED else None
        rows.append({
            "id": r.id,
            "status": r.status,
            "run": r.label,
            "progress": (p.get("n_done", 0) / p["n_tasks"]) if p.get("n_tasks") else 0.0,
            "cells": p.get("n_tasks"),
            "errors": p.get("n_errors", 0),
            "accuracy": report.get("accuracy") if report else None,
            "cost": p.get("cost_usd"),
            "duration": fmt_duration(r.elapsed_s),
            "created": datetime.fromtimestamp(r.created_at),
        })
    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch", key="history",
        on_select="rerun", selection_mode="single-row",
        column_config={
            "progress": st.column_config.ProgressColumn("Progress", min_value=0.0, max_value=1.0, format="percent"),
            "cells": st.column_config.NumberColumn("Cells", format="localized"),
            "accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
            "cost": st.column_config.NumberColumn("Cost", format="dollar"),
            "created": st.column_config.DatetimeColumn("Created", format="D MMM HH:mm"),
        },
    )
    selection = st.session_state.get("history", {}).get("selection", {}).get("rows", [])
    if selection:
        run_detail(runs[selection[0]])
    else:
        st.caption("Select a run for details.")


def run_detail(run) -> None:
    p = run.progress
    st.markdown(f"#### {run.label} · `{run.id}`")
    b = st.columns([1, 1, 4])
    if run.status in FINISHED and (run.status != DONE or p.get("n_errors")):
        label = "Retry failed cells" if run.status == DONE else "Resume"
        if b[0].button(label, icon=":material/replay:", key=f"resume_{run.id}",
                       help="Queue it again; cells already done are skipped."):
            store.requeue(run.id)
            st.rerun()
    if run.error:
        st.error(run.error)

    ticks = store.ticks(run.id)
    if len(ticks) >= 3:
        df = pd.DataFrame(ticks)
        df["time"] = pd.to_datetime(df["t"], unit="s")
        df["cells per minute"] = df["done"].diff() / df["t"].diff() * 60
        df = df.set_index("time")
        # 30-second rolling mean: ticks are 2 s apart and too spiky to read raw
        st.caption("Throughput (cells per minute)")
        st.line_chart(df["cells per minute"].rolling("30s").mean(), height=200, y_label="cells / min")

    c1, c2 = st.columns(2)
    with c1:
        st.caption("Counters")
        keys = ["n_tasks", "n_done", "n_ok", "n_empty", "n_errors", "n_resumed", "llm_calls", "retries",
                "rate_limited", "summaries_done", "summary_errors", "lamapi_cached", "lamapi_fetched",
                "lamapi_failed", "gt_cells", "gt_present", "input_tokens", "output_tokens", "reasoning_tokens",
                "cost_usd", "latency_p50_s", "latency_p90_s"]
        st.dataframe(pd.DataFrame({"value": [p.get(k) for k in keys]}, index=keys), width="stretch", height=420)
    with c2:
        st.caption("Configuration")
        st.json(run.config, expanded=False)
        if run.run_dir:
            st.caption(f"Outputs: `{run.run_dir}`")
        report = read_report(run)
        if report:
            st.caption("Evaluation")
            st.json(report, expanded=False)
    if p.get("gt_missing_examples"):
        st.warning(
            "Cells whose GT id is **not** among the candidates even though it was forced "
            "(LamAPI does not know the id, or the GT is not a Wikidata id):\n\n"
            + "\n".join(f"- `{x}`" for x in p["gt_missing_examples"])
        )
    if p.get("recent_errors"):
        st.caption("Recent errors")
        st.code("\n".join(p["recent_errors"]), language=None, wrap_lines=True)


monitor()
