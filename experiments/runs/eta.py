"""How long will a run take?

Before a run, throughput is the slowest of three limits (Little's law for the first one):

* parallelism       ``max_concurrency / seconds_per_call``
* request rate cap  ``max_rpm / 60``                       (when set)
* token rate cap    ``max_tpm / 60 / tokens_per_call``     (when set)

LamAPI lookups that are not cached run in their own pool (``lamapi_concurrency / seconds_per_lookup``)
in parallel with the LLM calls, so the run takes as long as the slower of the two pipelines.

During a run the ETA uses the throughput actually measured over the last minute instead.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import pandas as pd

from experiments.cost.candidates import _plain
from experiments.cost.tasks import TaskSet
from experiments.forcing import load_gt_ids

# Used only when nothing better is known (no calibration, no OpenRouter latency stats).
DEFAULT_SECONDS_PER_CALL = 2.0
DEFAULT_SECONDS_PER_CALL_REASONING = 8.0
DEFAULT_SECONDS_PER_LOOKUP = 0.6


@dataclass
class DurationEstimate:
    seconds: float
    calls: int
    lookups_to_fetch: int
    llm_rate_per_s: float
    lamapi_rate_per_s: float
    bottleneck: str  # "parallelism" | "rpm cap" | "tpm cap" | "LamAPI"
    seconds_per_call: float
    seconds_per_call_source: str
    max_concurrency: int
    max_rpm: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_duration(
    calls: int,
    seconds_per_call: float,
    max_concurrency: int,
    max_rpm: int = 0,
    max_tpm: int = 0,
    tokens_per_call: float = 0.0,
    lookups_to_fetch: int = 0,
    seconds_per_lookup: float = DEFAULT_SECONDS_PER_LOOKUP,
    lamapi_concurrency: int = 8,
    seconds_per_call_source: str = "assumed",
) -> DurationEstimate:
    limits = {"parallelism": max_concurrency / max(seconds_per_call, 1e-3)}
    if max_rpm > 0:
        limits["rpm cap"] = max_rpm / 60.0
    if max_tpm > 0 and tokens_per_call > 0:
        limits["tpm cap"] = max_tpm / 60.0 / tokens_per_call
    bottleneck = min(limits, key=limits.get)
    llm_rate = limits[bottleneck]
    lamapi_rate = lamapi_concurrency / max(seconds_per_lookup, 1e-3)

    llm_s = calls / llm_rate if calls else 0.0
    lamapi_s = lookups_to_fetch / lamapi_rate if lookups_to_fetch else 0.0
    if lamapi_s > llm_s:
        bottleneck = "LamAPI"
    # + one call latency: the pipeline has to fill before it runs at full speed, and the last
    # requests still take a full call to come back.
    seconds = max(llm_s, lamapi_s) + (seconds_per_call if calls else 0.0)
    return DurationEstimate(
        seconds=seconds,
        calls=calls,
        lookups_to_fetch=lookups_to_fetch,
        llm_rate_per_s=llm_rate,
        lamapi_rate_per_s=lamapi_rate,
        bottleneck=bottleneck,
        seconds_per_call=seconds_per_call,
        seconds_per_call_source=seconds_per_call_source,
        max_concurrency=max_concurrency,
        max_rpm=max_rpm,
    )


def default_seconds_per_call(
    reasoning: bool,
    calibrated: float | None = None,
    latency_s: float | None = None,
    throughput_tps: float | None = None,
    output_tokens: float | None = None,
) -> tuple[float, str]:
    """Best available guess for one call's latency, and where it comes from."""
    if calibrated:
        return calibrated, "measured by calibration"
    if latency_s and throughput_tps and output_tokens:
        return latency_s + output_tokens / throughput_tps, "OpenRouter provider stats (last 30 min)"
    if latency_s:
        return latency_s, "OpenRouter provider latency (last 30 min)"
    if reasoning:
        return DEFAULT_SECONDS_PER_CALL_REASONING, "assumed (reasoning model)"
    return DEFAULT_SECONDS_PER_CALL, "assumed"


def lookup_plan(
    taskset: TaskSet, client, force_gt: bool = True, force_ids: Iterable[str] = (), sample_size: int = 3000,
    seed: int = 13,
) -> dict[str, Any]:
    """Which LamAPI lookups of a run are already cached (same keys the executor will use)?

    ``client`` is a LamapiClient (offline is fine: only ``is_cached`` is used). Datasets with more
    than ``sample_size`` cells are sampled (a few whole tables' worth of reads instead of all of
    them) and the counts extrapolated; ``exact`` says which.
    """
    gt = load_gt_ids(taskset.source) if force_gt else {}
    extra = list(force_ids)
    tasks = taskset.tasks
    exact = len(tasks) <= sample_size
    if not exact:
        tasks = tasks.sample(sample_size, random_state=seed).sort_values(["table_id", "row", "col"])
    seen: set[tuple] = set()
    cached = to_fetch = empty = cells = 0
    for t in tasks.itertuples(index=False):
        df = taskset.load_table(t.table_id)
        if t.row >= len(df) or t.col >= len(df.columns):
            continue
        mention = df.iloc[t.row, t.col]
        if pd.isna(mention) or str(mention).strip() == "":
            empty += 1
            continue
        cells += 1
        mention = _plain(mention)
        forced = sorted(set(gt.get((t.table_id, int(t.row), int(t.col)), []) + extra)) or None
        key = (repr(mention), tuple(forced or ()))
        if key in seen:
            continue
        seen.add(key)
        if client is not None and client.is_cached(mention, ids=forced, target_kg=taskset.source.target_kg):
            cached += 1
        else:
            to_fetch += 1
    plan = {"cells": cells, "empty": empty, "unique_lookups": len(seen), "cached": cached, "to_fetch": to_fetch}
    if not exact:
        # Duplicate mentions are rarer in a sample than in the whole dataset, so to_fetch is an upper bound.
        scale = len(taskset.tasks) / len(tasks)
        plan = {k: round(v * scale) for k, v in plan.items()}
    plan["exact"] = exact
    return plan


# --------------------------------------------------------------------------- live


def live_eta(progress: dict[str, Any], ticks: list[dict[str, Any]], window_s: float = 90.0) -> dict[str, Any]:
    """Throughput over the last ``window_s`` seconds of ticks, and the ETA it implies."""
    n_tasks = progress.get("n_tasks") or 0
    done = progress.get("n_done") or 0
    remaining = max(n_tasks - done, 0)
    rate = None
    if ticks:
        now = ticks[-1]["t"]
        recent = [t for t in ticks if t["t"] >= now - window_s]
        if len(recent) >= 2 and recent[-1]["t"] > recent[0]["t"]:
            rate = (recent[-1]["done"] - recent[0]["done"]) / (recent[-1]["t"] - recent[0]["t"])
    if not rate:
        # whole session average (excludes cells resumed from an earlier attempt)
        started = progress.get("session_started_at")
        session_done = done - (progress.get("session_done_at_start") or 0)
        if started and session_done > 0:
            rate = session_done / max(time.time() - started, 1e-3)
    eta_s = remaining / rate if rate else None
    projected_cost = None
    paid = (progress.get("n_ok") or 0) + (progress.get("n_errors") or 0)
    if paid and n_tasks:
        spent = progress.get("cost_usd") or 0.0
        projected_cost = spent + spent / paid * max(n_tasks - done, 0)
    return {"rate_per_s": rate, "eta_s": eta_s, "remaining": remaining, "projected_cost_usd": projected_cost}


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, s = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {s:02d}s"
    hours, m = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h {m:02d}m"
    days, h = divmod(hours, 24)
    return f"{days}d {h}h"


def measured_seconds_per_call(store, model: str, provider: str | None = None) -> tuple[float, int] | None:
    """Median per-call latency seen in earlier runs of this model (and provider, if pinned)."""
    values = []
    for run in store.list(500):
        cfg, p = run.config, run.progress
        if cfg.get("model") != model or not p.get("latency_p50_s"):
            continue
        if provider and cfg.get("provider") not in (None, provider):
            continue
        values.append(float(p["latency_p50_s"]))
    if not values:
        return None
    values.sort()
    return values[len(values) // 2], len(values)


def seconds_per_lookup_for(store) -> tuple[float, str]:
    """Seconds per uncached LamAPI lookup measured in earlier runs, else the default."""
    total_s = total_n = 0.0
    if store is not None:
        for run in store.list(500):
            p = run.progress
            if p.get("lamapi_fetched"):
                total_s += p.get("lamapi_seconds") or 0.0
                total_n += p["lamapi_fetched"]
    if total_n >= 20:
        return total_s / total_n, f"measured on {int(total_n):,} lookups"
    return DEFAULT_SECONDS_PER_LOOKUP, "assumed"


def seconds_per_call_for(
    store, model: str, provider: str | None, reasoning: bool, calibrated: float | None = None,
    latency_s: float | None = None, throughput_tps: float | None = None, output_tokens: float | None = None,
) -> tuple[float, str]:
    """Calibration > earlier runs > OpenRouter stats > assumption."""
    if calibrated:
        return calibrated, "measured by calibration"
    past = measured_seconds_per_call(store, model, provider) if store is not None else None
    if past:
        return past[0], f"median of {past[1]} earlier run(s)"
    return default_seconds_per_call(reasoning, None, latency_s, throughput_tps, output_tokens)


def seconds_per_cell_for_method(store, method: str) -> tuple[float, int] | None:
    """Median seconds per cell of earlier finished runs of a classic linker (whole pipeline, per table)."""
    values = []
    if store is None:
        return None
    for run in store.list(500):
        p = run.progress
        if run.config.get("method") != method or run.status != "done" or not run.elapsed_s:
            continue
        done = (p.get("n_done") or 0) - (p.get("n_resumed") or 0)
        if done > 0:
            values.append(run.elapsed_s / done)
    if not values:
        return None
    values.sort()
    return values[len(values) // 2], len(values)
