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
from experiments.cost.tasks import datasets_root_problem, discover_sources, load_taskset
from experiments.runs.config import LINKER_METHODS, METHODS, RunConfig
from experiments.runs.byok import known_byok_providers, provider_preferences
from experiments.runs.metrics import run_metrics
from experiments.runs.eta import (
    estimate_duration, fmt_duration, live_eta, lookup_plan, seconds_per_call_for, seconds_per_cell_for_method,
    seconds_per_lookup_for,
)
from experiments.runs.store import ACTIVE, CANCELLING, DONE, FINISHED, QUEUED, RUNNING, RunStore

SCHEMAS = {"base": "Base", "structured": "Structured (detailed)", "fewshot": "Few-shot"}
DEFAULT_MODEL = "openai/gpt-oss-120b"
BYOK_ONLY, BYOK_FIRST = "Your BYOK providers only", "BYOK first, then shared capacity"
ROUTING, CHEAPEST = "OpenRouter routing", "Cheapest healthy provider (pinned)"
POLICY = {BYOK_ONLY: "byok", BYOK_FIRST: "byok_first", ROUTING: "routing", CHEAPEST: "pin"}
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


@st.cache_data(ttl=600, show_spinner=False)
def run_source(dataset: str, datasets_root: str | None):
    return RunConfig(dataset=dataset, model="-", datasets_root=datasets_root).resolve_source()


@st.cache_data(ttl=15, show_spinner=False)
def live_metrics(run_dir: str, dataset: str, datasets_root: str | None) -> dict | None:
    """P/R/F1 over the cells finished so far; recomputed at most every 15 s."""
    try:
        return run_metrics(run_dir, run_source(dataset, datasets_root))
    except Exception:  # noqa: BLE001 - unknown dataset, unreadable GT...
        return None


@st.cache_data(show_spinner=False)
def _final_metrics(run_dir: str, dataset: str, datasets_root: str | None, mtime: float) -> dict | None:
    path = Path(run_dir) / "report.json"
    try:
        report = json.loads(path.read_text()) if path.exists() else None
    except (OSError, json.JSONDecodeError):
        report = None
    if report and "f1" in report:
        return report
    # Older run without P/R/F1 in its report: score its results now.
    try:
        return run_metrics(run_dir, run_source(dataset, datasets_root)) or report
    except Exception:  # noqa: BLE001
        return report


def read_report(run) -> dict | None:
    """Final metrics of a finished run (report.json, or computed from results.jsonl)."""
    if not run.run_dir or not Path(run.run_dir).exists():
        return None
    results = Path(run.run_dir) / "results.jsonl"
    mtime = results.stat().st_mtime if results.exists() else 0.0
    return _final_metrics(run.run_dir, run.config.get("dataset", ""), run.config.get("datasets_root"), mtime)


def pct(x: float | None) -> str:
    return f"{x:.1%}" if x is not None else "—"


store = get_store()
st.title("Runs")

# --------------------------------------------------------------------------- launch

all_runs = store.list(300)
with st.expander("Queue new runs", icon=":material/add_circle:", expanded=not all_runs):
    datasets_root = os.getenv("DATASETS_ROOT", "")
    sources = get_sources(datasets_root)
    if problem := datasets_root_problem(datasets_root):
        st.warning(problem, icon=":material/folder_off:")
    else:
        n_semtab = sum(1 for s in sources if s.kind == "semtab")
        st.caption(f":material/folder: {n_semtab} SemTab dataset(s) from `{datasets_root}` + the repo datasets")
    by_key = {s.key: s for s in sources}
    prices = get_prices()

    c1, c2 = st.columns([3, 2])
    dataset_keys = c1.multiselect(
        "Datasets", list(by_key), default=[k for k in ["spec:mv"] if k in by_key],
        format_func=lambda k: f"{by_key[k].name}  ·  {'repo' if by_key[k].kind == 'spec' else 'SemTab'}",
    )
    methods = c2.multiselect(
        "Methods", list(METHODS), default=["llm"], format_func=METHODS.get,
        help="LLM: LionLinker prompts sent to OpenRouter models. Crocodile and Alligator: the classic ML table "
        "linkers, run by the `linkers` worker (MongoDB + TensorFlow image). All use the same LamAPI candidates "
        "and GT forcing, and are scored with the same P/R/F1.",
    )
    use_llm = "llm" in methods
    linkers = [m for m in methods if m in LINKER_METHODS]
    schemas, models, provider_mode, byok = [], [], BYOK_ONLY, set()
    if use_llm:
        st.markdown("**LLM**")
        c1, c2 = st.columns([2, 3])
        schemas = c1.multiselect("Prompt schemas", list(SCHEMAS), default=["base"], format_func=SCHEMAS.get)
        models = c2.multiselect(
            "Models (OpenRouter)", sorted(prices.models) or [DEFAULT_MODEL],
            default=[DEFAULT_MODEL] if DEFAULT_MODEL in prices.models or not prices.models else [],
            placeholder="Type to search, e.g. gpt-oss, llama, qwen...",
        )
        byok, byok_source = known_byok_providers(store)
        provider_mode = st.radio(
            "Provider", [BYOK_ONLY, BYOK_FIRST, ROUTING, CHEAPEST], horizontal=True,
            help="BYOK only: requests go to the providers you have your own key for, and a 429 makes the run back "
            "off and retry instead of moving to another provider - one provider and quantization per run, no "
            "OpenRouter credits. BYOK first: OpenRouter may fall back to shared capacity when your key is throttled. "
            "Routing: OpenRouter decides. Pinned: one specific endpoint.",
        )
        if byok:
            st.caption(f":material/key: BYOK providers: **{', '.join(sorted(byok))}** ({byok_source})")
        elif provider_mode in (BYOK_ONLY, BYOK_FIRST):
            st.caption(
                ":material/key_off: No BYOK provider known yet. " + (
                    "**BYOK only** runs will refuse to start; " if provider_mode == BYOK_ONLY else ""
                ) + "set `OPENROUTER_BYOK_PROVIDERS=cerebras,...` in .env (they are also learned from earlier "
                "runs, via OpenRouter's `is_byok` flag)."
            )

    c1, c2 = st.columns(2)
    limit = c1.number_input("Limit (first N rows / cells, 0 = all)", min_value=0, value=0, step=100)
    lamapi_limit = c2.number_input("Candidates per mention", min_value=1, max_value=100, value=10)
    # Inputs that apply to the chosen methods only: LLM parallelism, LamAPI parallelism, linker processes.
    cols = iter(st.columns(int(use_llm) * 2 + 1 + int(bool(linkers))))
    concurrency = max_rpm = 0
    if use_llm:
        concurrency = next(cols).number_input(
            "Parallel LLM requests", min_value=1, max_value=512, value=16,
            help="OpenRouter requests in flight at once. OpenRouter has no fixed limit on paid models, but "
            "providers answer 429 when pushed too hard; the run then pauses all LLM requests and retries.",
        )
        max_rpm = next(cols).number_input(
            "Max LLM requests / min (0 = no cap)", min_value=0, value=0, step=60,
            help="Optional hard cap, e.g. your BYOK provider's documented requests-per-minute limit.",
        )
    lamapi_concurrency = next(cols).number_input(
        "Parallel LamAPI requests", min_value=1, max_value=128, value=8,
        help="LLM runs: candidate lookups in flight at once (only uncached ones hit LamAPI), alongside the "
        "LLM requests; LamAPI 429/5xx responses are retried with back-off. Crocodile/Alligator fetch "
        "candidates themselves, with their worker processes.",
    )
    linker_workers = 4
    if linkers:
        linker_workers = next(cols).number_input(
            "Crocodile / Alligator processes", min_value=1, max_value=32, value=4,
            help="Worker processes each linker uses per table (candidate retrieval and ML ranking).",
        )

    c1, c2 = st.columns([1, 2])
    force_gt = c1.checkbox(
        "Force GT candidates", value=True,
        help="Every GT id of a cell is passed to LamAPI (`ids`, space-separated), so the correct entity is "
        "always among the candidates. The run reports how many cells still miss it.",
    )
    extra_ids = c2.text_input("Extra ids forced into every candidate list", placeholder="Q42 Q64")
    force_ids = tuple(extra_ids.replace(",", " ").split())

    combos = [(d, s, m) for d in dataset_keys for s in schemas for m in models]
    linker_combos = [(d, meth) for d in dataset_keys for meth in linkers]
    spl, spl_src = seconds_per_lookup_for(store)

    # ---- time estimate
    if combos or linker_combos:
        rows = []
        for d, meth in linker_combos:
            plan = get_plan(d, datasets_root, int(limit), int(lamapi_limit), force_gt, force_ids)
            measured = seconds_per_cell_for_method(store, meth)
            seconds = plan["cells"] * measured[0] if measured else None
            rows.append({
                "dataset": by_key[d].name, "method": METHODS[meth], "schema": "—", "model": "—", "provider": "—",
                "calls": plan["cells"], "uncached lookups": None, "s/call": round(measured[0], 3) if measured else None,
                "counts": "exact" if plan.get("exact", True) else "sampled",
                "latency source": f"median of {measured[1]} earlier run(s), per cell" if measured
                else "unknown until the first run",
                "bottleneck": "—", "duration": fmt_duration(seconds) if seconds else "measured by the first run",
                "_seconds": seconds or 0.0, "_cfg": RunConfig(
                    dataset=d, method=meth, limit=int(limit), lamapi_limit=int(lamapi_limit), force_gt=force_gt,
                    force_ids=list(force_ids), linker_workers=int(linker_workers),
                    datasets_root=datasets_root or None,
                ), "_est": None,
            })
        for d, s, m in combos:
            plan = get_plan(d, datasets_root, int(limit), int(lamapi_limit), force_gt, force_ids)
            price = prices.models.get(m)
            pinned = None
            latency = throughput = None
            eps = get_endpoints(m)
            if provider_mode == CHEAPEST:
                best = eps.cheapest(700, 400 if price and price.reasoning else 10)
                if best:
                    pinned, latency, throughput = best.tag, best.latency_s, best.throughput_tps
            policy = "pin" if pinned else ("routing" if provider_mode == CHEAPEST else POLICY[provider_mode])
            _, route = provider_preferences(policy, pinned, eps, byok)
            spc, spc_src = seconds_per_call_for(
                store, m, pinned, bool(price and price.reasoning), latency_s=latency, throughput_tps=throughput,
                output_tokens=400 if price and price.reasoning else 10,
            )
            calls = plan["cells"] + plan["tables"]
            est = estimate_duration(
                calls, spc, int(concurrency), int(max_rpm), lookups_to_fetch=plan["to_fetch"],
                seconds_per_lookup=spl, lamapi_concurrency=int(lamapi_concurrency),
                seconds_per_call_source=spc_src,
            )
            rows.append({
                "dataset": by_key[d].name, "method": METHODS["llm"], "schema": s, "model": m, "provider": route,
                "calls": calls, "uncached lookups": plan["to_fetch"], "s/call": round(spc, 2),
                "counts": "exact" if plan.get("exact", True) else "sampled",
                "latency source": spc_src, "bottleneck": est.bottleneck, "duration": fmt_duration(est.seconds),
                "_seconds": est.seconds, "_cfg": RunConfig(
                    dataset=d, model=m, schema=s, provider=pinned, provider_policy=policy,
                    limit=int(limit), lamapi_limit=int(lamapi_limit),
                    force_gt=force_gt, force_ids=list(force_ids), max_concurrency=int(concurrency),
                    max_rpm=int(max_rpm), lamapi_concurrency=int(lamapi_concurrency),
                    datasets_root=datasets_root or None,
                ), "_est": est,
            })
        table = pd.DataFrame(rows)
        total_s = table["_seconds"].sum()
        unmeasured = sum(1 for r in rows if r["_seconds"] == 0 and r["_est"] is None)
        if unmeasured == len(rows):
            st.markdown(f"**{len(rows)} run(s)** - duration unknown until a first Crocodile/Alligator run is measured.")
        else:
            st.markdown(
                f"**{len(rows)} run(s), about {fmt_duration(total_s)}** if each worker runs them one after another"
                + (f" (plus {unmeasured} linker run(s) not measured yet)" if unmeasured else "") + "."
            )
        st.dataframe(
            table.drop(columns=["_seconds", "_cfg", "_est"]), hide_index=True, width="stretch",
            column_config={
                "calls": st.column_config.NumberColumn(
                    "LLM calls / cells", format="localized",
                    help="LLM: non-empty mention cells + one table summary per table. Crocodile/Alligator: cells."),
                "uncached lookups": st.column_config.NumberColumn(
                    format="localized", help="LamAPI lookups not in data/lamapi_cache yet (fetched during the run)."),
                "s/call": st.column_config.NumberColumn("Seconds / call"),
                "bottleneck": st.column_config.TextColumn(
                    help="parallelism = parallel requests ÷ seconds per call; rpm cap = your requests/min cap; "
                    "LamAPI = fetching uncached candidates is slower than the LLM calls."),
            },
        )
        st.caption(
            ("Crocodile/Alligator duration = cells × the median seconds per cell of their earlier finished runs "
             "(whole pipeline, one table at a time). " if linkers else "") +
            ("" if not use_llm else
            "LLM duration = the slower of two pipelines running side by side: LLM calls ÷ min(*parallel LLM requests ÷ "
            "seconds per call*, requests/min cap), and uncached lookups ÷ (*parallel LamAPI requests ÷ seconds per "
            f"lookup*, {spl:.2f}s {spl_src}). Retries after 429s are not included. Seconds per call come from "
            "calibration or earlier runs of the same model when available. For cost, use the **Estimate cost** page.")
        )
        if linkers and not any(set(linkers) & set((w.get("methods") or "llm").split(",")) for w in store.workers()):
            st.warning(
                "No worker for Crocodile/Alligator is running - those runs will wait. Start it with "
                "`docker compose up -d linkers` (first build ~1.8 GB).", icon=":material/warning:",
            )
        if use_llm and not os.getenv("OPENROUTER_API_KEY"):
            st.error("OPENROUTER_API_KEY is not set in .env - runs would fail at once.")
        if st.button(f"Queue {len(rows)} run(s)", type="primary", icon=":material/play_arrow:"):
            for r in rows:
                cfg = r["_cfg"]
                store.submit(cfg.to_dict(), cfg.label(), estimate=r["_est"].to_dict() if r["_est"] else {})
            st.toast(f"Queued {len(rows)} run(s).", icon=":material/check:")
            st.rerun()
    else:
        st.info("Pick at least one dataset and method (LLM runs also need a prompt schema and a model).")


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
        is_llm = run.config.get("method", "llm") == "llm"
        if is_llm:
            m[2].metric("Cost so far", usd(p.get("cost_usd")), help=f"Projected total: {usd(eta.get('projected_cost_usd'))}")
            m[3].metric(
                "In flight", f"{p.get('in_flight', 0)} / {p.get('max_concurrency', '?')}",
                help="Requests in flight / max parallel requests" + (" · paused after a 429" if p.get("cooling_down") else ""),
            )
            m[4].metric("Errors · 429s", f"{p.get('n_errors', 0)} · {p.get('rate_limited', 0)}",
                        help=f"{p.get('retries', 0)} retries so far")
        else:
            m[2].metric("Tables", f"{p.get('tables_done', 0):,} / {p.get('tables_total', 0):,}",
                        help="Tables linked in this attempt (cells finish a table at a time).")
            m[3].metric("Cells in current table", f"{p.get('in_flight', 0):,}")
            m[4].metric("Errors", f"{p.get('n_errors', 0):,}", help="Cells of tables the linker failed on.")
        gt_cells = p.get("gt_cells") or 0
        m[5].metric("GT in candidates", f"{p.get('gt_present', 0) / gt_cells:.1%}" if gt_cells else "—",
                    help="Cells whose GT id is among the candidates sent to the model.")
        live = live_metrics(run.run_dir, run.config.get("dataset", ""), run.config.get("datasets_root")) \
            if run.run_dir else None
        byok_calls, shared = p.get("byok_calls", 0), p.get("shared_calls", 0)
        if not is_llm and p.get("provider_route"):
            st.caption(f":material/account_tree: {p['provider_route']}")
        elif p.get("provider_route") or byok_calls or shared:
            share = f" · {byok_calls / (byok_calls + shared):.0%} of calls on your keys" if byok_calls + shared else ""
            warn = (
                " :orange[- some calls ran on OpenRouter's shared capacity]" if shared and byok_calls else ""
            )
            if p.get("shared_rejected"):
                warn += (
                    f" :orange[- {p['shared_rejected']:,} answers came from shared capacity and were discarded "
                    f"and retried ({usd(p.get('shared_rejected_usd'))} wasted); set your BYOK key to *Never use "
                    "shared capacity* in OpenRouter to avoid paying for them]"
                )
            st.caption(f":material/key: {p.get('provider_route', '')}{share}{warn}")
        if live:
            st.caption(
                f":material/insights: So far on {live['evaluated']:,} cells with GT - "
                f"**P {pct(live['precision'])} · R {pct(live['recall'])} · F1 {pct(live['f1'])}** · "
                f"accuracy {pct(live['accuracy'])}"
            )
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
            "precision": report.get("precision") if report else None,
            "recall": report.get("recall") if report else None,
            "f1": report.get("f1") if report else None,
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
            "precision": st.column_config.NumberColumn(
                "P", format="percent", help="Correct entity predictions / cells where an entity was predicted."),
            "recall": st.column_config.NumberColumn(
                "R", format="percent", help="Correct entity predictions / cells whose GT is an entity."),
            "f1": st.column_config.NumberColumn("F1", format="percent"),
            "accuracy": st.column_config.NumberColumn(
                "Acc", format="percent", help="All cells right, NIL included."),
            "cost": st.column_config.NumberColumn("Cost", format="dollar"),
            "created": st.column_config.DatetimeColumn("Created", format="D MMM HH:mm"),
        },
    )
    selection = st.session_state.get("history", {}).get("selection", {}).get("rows", [])
    if selection:
        run_detail(runs[selection[0]])
    else:
        st.caption("Select a run for details.")
    compare(runs, rows)


def compare(runs, rows) -> None:
    """F1 of the latest finished run per configuration, one chart per dataset."""
    scored = [
        (r, row) for r, row in zip(runs, rows)
        if r.status in FINISHED and row["f1"] is not None
    ]
    if len(scored) < 2:
        return
    st.subheader("Compare")
    df = pd.DataFrame([
        {"dataset": r.config.get("dataset", "").split(":", 1)[-1], "id": r.id,
         # the dataset is the chart heading, so bars are labelled by prompt and model only
         "run": METHODS.get(r.config.get("method", "llm"), "?") if r.config.get("method", "llm") != "llm" else
                " · ".join(x for x in (r.config.get("schema"), r.config.get("model", "").split("/")[-1],
                                         r.config.get("provider")) if x),
         "F1": row["f1"], "P": row["precision"], "R": row["recall"], "cells": row["cells"],
         "cost": row["cost"], "created": r.created_at}
        for r, row in scored
    ]).sort_values("created").drop_duplicates("run", keep="last")
    st.caption("Latest finished run per dataset · prompt · model. Bars show F1; hover for the value.")
    for dataset, part in df.groupby("dataset", sort=True):
        part = part.sort_values("F1", ascending=False)
        st.markdown(f"**{dataset}**")
        if part["cells"].nunique() > 1:
            st.caption(
                ":material/warning: These runs cover different numbers of cells "
                f"({', '.join(f'{int(c):,}' for c in sorted(part['cells'].dropna().unique()))}); "
                "compare F1 only between runs with the same limit."
            )
        # a single series: the chart title names it, no legend needed
        st.bar_chart(part.set_index("run")["F1"], horizontal=True, height=60 + 32 * len(part),
                     x_label="", y_label="F1")
        st.dataframe(
            part[["run", "P", "R", "F1", "cells", "cost", "id"]], hide_index=True, width="stretch",
            column_config={
                "P": st.column_config.NumberColumn(format="percent"),
                "R": st.column_config.NumberColumn(format="percent"),
                "F1": st.column_config.NumberColumn(format="percent"),
                "cells": st.column_config.NumberColumn("Cells", format="localized"),
                "cost": st.column_config.NumberColumn("Cost", format="dollar"),
            },
        )


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

    report = read_report(run) if run.status in FINISHED else (
        live_metrics(run.run_dir, run.config.get("dataset", ""), run.config.get("datasets_root"))
        if run.run_dir else None
    )
    if report:
        scope = "" if run.status == DONE else f" (on the {report['evaluated']:,} cells finished so far)"
        st.caption(f"Linking quality{scope}")
        m = st.columns(4)
        m[0].metric("Precision", pct(report["precision"]),
                    help=f"{report['tp']:,} correct / {report['predicted_entities']:,} entity predictions")
        m[1].metric("Recall", pct(report["recall"]),
                    help=f"{report['tp']:,} correct / {report['gold_entities']:,} cells whose GT is an entity")
        m[2].metric("F1", pct(report["f1"]))
        m[3].metric("Accuracy", pct(report["accuracy"]),
                    help=f"{report['correct']:,} / {report['evaluated']:,} cells right, NIL included")
        if report.get("nil_gt_count"):
            st.caption(
                f"NIL detection ({report['nil_gt_count']:,} GT NIL cells): P {pct(report['nil_precision'])} · "
                f"R {pct(report['nil_recall'])} · F1 {pct(report['nil_f1'])}"
            )
    elif run.status in FINISHED:
        st.caption("No GT for this dataset, so no P/R/F1.")

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
                "cost_usd", "latency_p50_s", "latency_p90_s", "byok_calls", "shared_calls", "shared_rejected",
                "shared_rejected_usd"]
        st.dataframe(pd.DataFrame({"value": [p.get(k) for k in keys]}, index=keys), width="stretch", height=420)
    with c2:
        st.caption("Configuration")
        st.json(run.config, expanded=False)
        if run.run_dir:
            st.caption(f"Outputs: `{run.run_dir}`")
        if report:
            st.caption("Evaluation report")
            st.json(report, expanded=False)
    if p.get("providers"):
        st.caption("Calls per provider: " + " · ".join(
            f"{name} {n:,}" for name, n in sorted(p["providers"].items(), key=lambda x: -x[1])))
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
