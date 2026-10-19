"""Estimate page: pick datasets x prompt schemas x models, get tokens and USD before running."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import streamlit as st

from experiments.cost.calibrate import calibrate
from experiments.cost.candidates import check_lamapi
from experiments.cost.estimate import (
    DEFAULT_OUTPUT_TOKENS,
    DEFAULT_REASONING_OUTPUT_TOKENS,
    DEFAULT_SUMMARY_TOKENS,
    ProfileConfig,
    make_lamapi_clients,
    price_profile,
    profile_prompts,
    profile_to_dict,
)
from experiments.cost.pricing import PriceBook, ProviderList, load_endpoints, load_prices, with_provider
from experiments.cost.tasks import REPO_ROOT, discover_sources, load_taskset
from experiments.cost.tokens import get_counter
from experiments.runs.eta import estimate_duration, fmt_duration, seconds_per_call_for
from experiments.runs.store import RunStore

SCHEMAS = {
    "base": "Base",
    "structured": "Structured (detailed)",
    "fewshot": "Few-shot",
}
DEFAULT_MODEL = "openai/gpt-oss-120b"
CHEAPEST = "__cheapest__"
TYPICAL_INPUT = 700  # tokens; only used to rank providers


# --------------------------------------------------------------------------- cached loaders


@st.cache_data(ttl=300, show_spinner=False)
def get_sources(root: str):
    return discover_sources(Path(root).expanduser() if root else None)


@st.cache_data(show_spinner="Reading dataset...")
def get_taskset(source_key: str, root: str, limit: int):
    source = next(s for s in get_sources(root) if s.key == source_key)
    return load_taskset(source, limit)


@st.cache_data(ttl=3600, show_spinner="Loading OpenRouter prices...")
def get_prices(refresh_token: int) -> PriceBook:
    return load_prices(refresh=refresh_token > 0)


@st.cache_data(ttl=3600, show_spinner="Loading providers...")
def get_endpoints(model_id: str, refresh_token: int) -> ProviderList:
    return load_endpoints(model_id, refresh=refresh_token > 0)


@st.cache_data(ttl=300, show_spinner=False)
def get_lamapi_status():
    return check_lamapi()


@st.cache_resource
def get_store() -> RunStore:
    return RunStore()


@st.cache_resource
def get_lamapi(limit: int):
    return make_lamapi_clients(limit)


def usd(x: float) -> str:
    if x == 0:
        return "$0"
    if x < 0.01:
        return "<$0.01"
    return f"${x:,.2f}"


def big(n: float) -> str:
    n = float(n)
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.1f}{unit}"
    return f"{n:.0f}"


st.session_state.setdefault("profiles", {})  # (dataset, root, limit, cfg) -> PromptProfile
st.session_state.setdefault("price_refresh", 0)
st.session_state.setdefault("calibrations", {})  # (profile key, model, provider) -> Calibration

# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Settings")
    datasets_root = st.text_input(
        "Datasets folder",
        value=os.getenv("DATASETS_ROOT", ""),
        help="Shared folder with SemTab-style datasets (set DATASETS_ROOT in .env to make this the default).",
    )
    prices = get_prices(st.session_state.price_refresh)
    if prices.source == "unavailable":
        st.error(f"OpenRouter prices unavailable: {prices.error}")
    else:
        when = datetime.fromtimestamp(prices.fetched_at).strftime("%d %b %H:%M")
        label = "live" if prices.source == "live" else "cached"
        st.caption(f"Prices: {len(prices.models)} OpenRouter models ({label}, {when})")
    if st.button("Refresh prices", icon=":material/refresh:"):
        st.session_state.price_refresh += 1
        st.rerun()

    offline_client, _ = get_lamapi(10)
    lamapi_ok, lamapi_msg = get_lamapi_status()
    n_cached = sum(1 for _ in Path(offline_client.cache_dir).glob("*.json"))
    host = urlparse(offline_client.endpoint).netloc
    if lamapi_ok:
        st.caption(f"LamAPI: {host} reachable · {n_cached:,} cached lookups")
    else:
        st.warning(f"LamAPI unreachable - only cached candidates ({n_cached:,}) can be used. {lamapi_msg}")
    st.caption(f"Tokenizer: {get_counter().name}")

# --------------------------------------------------------------------------- 1. selection

st.title("Estimate cost")
st.caption(
    "Builds the exact prompts LionLinker would send, counts their tokens and prices them with "
    "OpenRouter rates. Nothing is sent to an LLM."
)

sources = get_sources(datasets_root)
source_by_key = {s.key: s for s in sources}

st.subheader("1. What do you want to run?")
c1, c2 = st.columns([3, 2])
with c1:
    dataset_keys = st.multiselect(
        "Datasets",
        options=list(source_by_key),
        default=[k for k in ["spec:mv"] if k in source_by_key],
        format_func=lambda k: f"{source_by_key[k].name}  ·  {'repo' if source_by_key[k].kind == 'spec' else 'SemTab'}",
    )
with c2:
    schemas = st.multiselect(
        "Prompt schemas", options=list(SCHEMAS), default=["base"], format_func=SCHEMAS.get
    )

model_ids = sorted(prices.models)
model_choices = st.multiselect(
    "Models (OpenRouter)",
    options=model_ids,
    default=[m for m in [DEFAULT_MODEL] if m in prices.models],
    format_func=lambda m: m + ("  ·  reasoning" if prices.models[m].reasoning else ""),
    placeholder="Type to search, e.g. gpt-oss, llama, qwen...",
)

selected = []  # (ModelPrice at the chosen provider, ProviderList | None)
if model_choices:
    st.markdown("**Provider**")
    st.caption(
        "OpenRouter serves each model through several providers, at different prices and quantizations. "
        "Unless a provider is pinned, OpenRouter routes between them, so the real cost can land anywhere "
        "in the *across providers* range shown in the results."
    )
for m in model_choices:
    base = prices.models[m]
    eps = get_endpoints(m, st.session_state.price_refresh)
    typical_out = st.session_state.get(
        "out_tok_reasoning" if base.reasoning else "out_tok",
        DEFAULT_REASONING_OUTPUT_TOKENS if base.reasoning else DEFAULT_OUTPUT_TOKENS,
    )
    if not eps.providers:
        st.caption(f"{m}: provider list unavailable ({eps.error}) - using the OpenRouter list price.")
        selected.append((base, None))
        continue
    ranked = sorted(eps.providers, key=lambda p: (not p.healthy, p.call_cost(TYPICAL_INPUT, typical_out)))
    cheapest = eps.cheapest(TYPICAL_INPUT, typical_out)

    def fmt(tag: str, eps=eps, cheapest=cheapest) -> str:
        p = cheapest if tag == CHEAPEST else eps.by_tag(tag)
        desc = (
            f"{p.provider} · {p.quantization} · ${p.input_per_token * 1e6:.3f} in / "
            f"${p.output_per_token * 1e6:.3f} out per 1M" + ("" if p.healthy else " · currently degraded")
        )
        return f"Cheapest available - {desc}" if tag == CHEAPEST else desc

    choice = st.selectbox(m, [CHEAPEST] + [p.tag for p in ranked], format_func=fmt, key=f"provider_{m}")
    chosen = cheapest if choice == CHEAPEST else eps.by_tag(choice)
    selected.append((with_provider(base, chosen), eps))

models = [mp for mp, _ in selected]
provider_lists = {mp.id: eps for mp, eps in selected}

# --------------------------------------------------------------------------- 2. assumptions

with st.expander("2. Assumptions (defaults match the Runs page and `make run-llm`)", icon=":material/tune:"):
    a1, a2, a3 = st.columns(3)
    with a1:
        limit = st.number_input(
            "Limit: first N rows / cells per dataset", min_value=0, value=0, step=100,
            help="0 = whole dataset. Same as --max-rows for the repo datasets.",
        )
        lamapi_limit = st.number_input("Candidates per mention (LamAPI limit)", min_value=1, max_value=100, value=10)
    with a2:
        out_tokens = st.number_input(
            "Output tokens per call - normal models", min_value=1, value=DEFAULT_OUTPUT_TOKENS, key="out_tok",
            help="Replaced by the measured value once a run is calibrated.",
        )
        out_tokens_reasoning = st.number_input(
            "Output tokens per call - reasoning models", min_value=1, value=DEFAULT_REASONING_OUTPUT_TOKENS,
            key="out_tok_reasoning",
            help="Hidden reasoning is billed as output. Replaced by the measured value once a run is calibrated.",
        )
    with a3:
        summary_tokens = st.number_input(
            "Table summary length (tokens)", min_value=0, value=DEFAULT_SUMMARY_TOKENS,
            help="LionLinker has the model write one summary per table and pastes it into every prompt. "
            "Replaced by the real summary once a run is calibrated.",
        )
        sample_size = st.number_input(
            "Prompts to build per dataset", min_value=100, value=1000, step=100,
            help="Smaller datasets are measured exactly; larger ones are sampled and extrapolated with a 95% range.",
        )
    fetch_missing = st.checkbox(
        "Fill prompts with real LamAPI candidates (free; also warms the cache for the real run)",
        value=True,
    )
    budget = st.number_input("Budget (USD, 0 = none)", min_value=0.0, value=0.0, step=10.0)
    force_gt = st.checkbox(
        "Force GT ids into the candidates (as runs do by default)", value=True,
        help="Counts the forced GT candidates in the prompts and warms the exact cache entries runs will use.",
    )
    t1, t2 = st.columns(2)
    concurrency = t1.number_input(
        "Parallel requests (for the duration estimate)", min_value=1, max_value=512, value=16,
        help="Same as the Runs page / --max-concurrency.",
    )
    max_rpm = t2.number_input("Max requests / min (0 = no cap)", min_value=0, value=0, step=60)

cfgs = {
    s: ProfileConfig(
        schema=s, sample_size=int(sample_size), lamapi_limit=int(lamapi_limit),
        fetch_missing=fetch_missing and lamapi_ok, force_gt=force_gt,
    )
    for s in schemas
}


def profile_key(dk: str, schema: str):
    return (dk, datasets_root, int(limit), cfgs[schema])


def cal_key(dk: str, schema: str, m) -> tuple:
    return (profile_key(dk, schema), m.id, m.provider)


wanted = [(dk, s) for dk in dataset_keys for s in schemas]
missing = [w for w in wanted if profile_key(*w) not in st.session_state.profiles]

ready = bool(dataset_keys and schemas and models)
if not ready:
    st.info("Pick at least one dataset, one prompt schema and one model.")

if st.button("Estimate cost", type="primary", icon=":material/calculate:", disabled=not ready or not missing):
    clients = get_lamapi(int(lamapi_limit))
    bar = st.progress(0.0, text="Starting...")
    for i, (dk, schema) in enumerate(missing):
        ts = get_taskset(dk, datasets_root, int(limit))

        def report(frac: float, msg: str, i=i, schema=schema) -> None:
            bar.progress(min((i + frac) / len(missing), 1.0), text=f"{msg}  ({SCHEMAS[schema]})")

        st.session_state.profiles[profile_key(dk, schema)] = profile_prompts(ts, cfgs[schema], clients, report)
    bar.empty()
    st.rerun()

if ready and missing and len(missing) < len(wanted):
    st.warning("Selection changed - click **Estimate cost** to update the numbers below.")

# --------------------------------------------------------------------------- 3. results

done = [w for w in wanted if profile_key(*w) in st.session_state.profiles]
if not (ready and done and not missing):
    st.stop()
profiles = [st.session_state.profiles[profile_key(*w)] for w in done]

rows, row_meta = [], []
for (dk, schema), prof in zip(done, profiles):
    for m in models:
        per_call = int(out_tokens_reasoning if m.reasoning else out_tokens)
        cal = st.session_state.calibrations.get(cal_key(dk, schema, m))
        row = price_profile(prof, m, per_call, int(summary_tokens), cal)
        row["reasoning"] = m.reasoning
        eps = provider_lists.get(m.id)
        healthy = [p for p in eps.providers if p.healthy] if eps else []
        if healthy:
            costs = [price_profile(prof, with_provider(m, p), per_call, int(summary_tokens), cal)["cost_usd"] for p in healthy]
            row["providers_min_usd"], row["providers_max_usd"] = min(costs), max(costs)
        else:
            row["providers_min_usd"] = row["providers_max_usd"] = row["cost_usd"]
        spc, spc_src = seconds_per_call_for(
            get_store(), m.id, m.provider, m.reasoning, calibrated=cal.seconds_per_call if cal and cal.ok else None,
            output_tokens=per_call,
        )
        # Lookups not cached yet: sampled ones were just fetched; extrapolate the miss rate to the rest.
        looked_up = prof.cache_hits + prof.fetched + prof.cache_misses
        miss_rate = (prof.fetched + prof.cache_misses) / looked_up if looked_up else 1.0
        to_fetch = prof.cache_misses + (0 if prof.exact else round((prof.n_tasks - prof.n_sampled) * miss_rate))
        est = estimate_duration(
            row["llm_calls"], spc, int(concurrency), int(max_rpm), lookups_to_fetch=to_fetch,
            seconds_per_call_source=spc_src,
        )
        row["duration_s"] = est.seconds
        row["duration"] = fmt_duration(est.seconds)
        row["seconds_per_call"] = spc
        row["seconds_per_call_source"] = spc_src
        row["bottleneck"] = est.bottleneck
        rows.append(row)
        row_meta.append((dk, schema, m, prof))
res = pd.DataFrame(rows)

st.subheader("3. Estimate")
total, low, high = res["cost_usd"].sum(), res["cost_low_usd"].sum(), res["cost_high_usd"].sum()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total cost", usd(total), help=f"95% range from sampling: {usd(low)} - {usd(high)}")
m2.metric("LLM calls", big(res["llm_calls"].sum()))
m3.metric("Input tokens", big(res["input_tokens"].sum()))
m4.metric("Output tokens", big(res["output_tokens"].sum()))
m5.metric(
    "Duration", fmt_duration(res["duration_s"].sum()),
    help=f"All runs one after another at {int(concurrency)} parallel requests.",
)
notes = []
if high > low * 1.001:
    notes.append(f"sampling range {usd(low)} - {usd(high)}")
pmin, pmax = res["providers_min_usd"].sum(), res["providers_max_usd"].sum()
if pmax > pmin * 1.001:
    notes.append(f"across healthy providers {usd(pmin)} - {usd(pmax)}")
if notes:
    st.caption("Total: " + "; ".join(notes) + ".")

if budget > 0:
    if total > budget:
        st.error(f"Over budget by {usd(total - budget)} ({total / budget:.0%} of {usd(budget)}).", icon=":material/warning:")
    else:
        st.success(f"Within budget: {total / budget:.0%} of {usd(budget)}.", icon=":material/check_circle:")

n_cal = int(res["calibrated"].sum())
if n_cal < len(res):
    st.info(
        f"{len(res) - n_cal} of {len(res)} runs still use assumptions for the table summary and output tokens"
        + (" (reasoning models: the biggest unknown)" if res["reasoning"].any() else "")
        + ". Calibrate them in step 4 for exact numbers.",
        icon=":material/science:",
    )

show = res.assign(
    schema=res["schema"].map(SCHEMAS),
    basis=res["calibrated"].map({True: "Calibrated", False: "Assumed"}),
    range=res.apply(lambda r: "exact" if r["exact"] else f"{usd(r['cost_low_usd'])} - {usd(r['cost_high_usd'])}", axis=1),
    across=res.apply(lambda r: f"{usd(r['providers_min_usd'])} - {usd(r['providers_max_usd'])}", axis=1),
)[["dataset", "schema", "model", "provider", "basis", "llm_calls", "input_tokens", "output_tokens", "cost_usd",
   "range", "across", "duration", "seconds_per_call", "bottleneck"]]
st.dataframe(
    show.sort_values("cost_usd", ascending=False),
    hide_index=True,
    width="stretch",
    column_config={
        "dataset": "Dataset",
        "schema": "Prompt",
        "model": "Model",
        "provider": "Provider",
        "basis": "Basis",
        "llm_calls": st.column_config.NumberColumn("LLM calls", format="localized"),
        "input_tokens": st.column_config.NumberColumn("Input tokens", format="localized"),
        "output_tokens": st.column_config.NumberColumn("Output tokens", format="localized"),
        "cost_usd": st.column_config.NumberColumn("Cost", format="dollar"),
        "range": "Sampling range",
        "across": "Across providers",
        "duration": st.column_config.TextColumn(
            "Duration",
            help="Calls ÷ throughput, with throughput = parallel requests ÷ seconds per call (or the requests/min "
            "cap if lower). Uncached LamAPI lookups run alongside. Retries after 429s not included.",
        ),
        "seconds_per_call": st.column_config.NumberColumn(
            "s / call", format="%.2f",
            help="From calibration, else the median of earlier runs of this model, else an assumption.",
        ),
        "bottleneck": "Bottleneck",
    },
)

if len(res) > 1:
    chart = res.assign(run=res["dataset"] + " · " + res["schema"] + " · " + res["model"].str.split("/").str[-1])
    st.bar_chart(chart.set_index("run")["cost_usd"], horizontal=True, y_label="", x_label="USD")

# --------------------------------------------------------------------------- 4. calibration

st.subheader("4. Calibrate with real calls")
st.caption(
    "For a few tables per run this does exactly what the run will do: the model writes the real table "
    "summary, then receives fully filled linking prompts (real LamAPI candidates + that summary). "
    "The provider's usage report then gives the real prompt tokens, output tokens (incl. reasoning) "
    "and summary length, which replace the assumptions above."
)
api_key_set = bool(os.getenv("OPENROUTER_API_KEY"))
k1, k2 = st.columns([1, 2])
with k1:
    n_cal_prompts = st.number_input("Real prompts per run", min_value=5, max_value=200, value=20, step=5)
todo = [i for i, r in enumerate(rows) if not r["calibrated"]]
planned_calls = sum(int(n_cal_prompts) + min(5, row_meta[i][3].n_tables) for i in todo)
planned_cost = sum(
    rows[i]["cost_usd"] / max(rows[i]["llm_calls"], 1) * (int(n_cal_prompts) + min(5, row_meta[i][3].n_tables))
    for i in todo
)
with k2:
    st.write("")
    clicked = st.button(
        f"Calibrate {len(todo)} run(s): {planned_calls} calls, about {usd(planned_cost)}" if todo else "All runs calibrated",
        icon=":material/science:",
        disabled=not todo or not api_key_set,
    )
if not api_key_set:
    st.caption("Set OPENROUTER_API_KEY in .env to enable calibration.")

if clicked:
    clients = get_lamapi(int(lamapi_limit))
    bar = st.progress(0.0, text="Calibrating...")
    for j, i in enumerate(todo):
        dk, schema, m, _ = row_meta[i]
        ts = get_taskset(dk, datasets_root, int(limit))

        def report(frac: float, msg: str, j=j) -> None:
            bar.progress(min((j + frac) / len(todo), 1.0), text=msg)

        st.session_state.calibrations[cal_key(dk, schema, m)] = calibrate(
            ts, schema, m, clients, n_prompts=int(n_cal_prompts), seed=cfgs[schema].seed, progress=report,
            force_gt=force_gt,
        )
    bar.empty()
    st.rerun()

cal_rows = [
    (i, st.session_state.calibrations.get(cal_key(dk, schema, m)))
    for i, (dk, schema, m, _) in enumerate(row_meta)
]
cal_rows = [(i, c) for i, c in cal_rows if c is not None]
if cal_rows:
    cal_df = pd.DataFrame(
        [
            {
                "Run": f"{rows[i]['dataset']} · {SCHEMAS[rows[i]['schema']]} · {c.model} @ {c.provider or 'auto'}",
                "Output tok / call": round(c.output_tokens_per_call),
                "of which reasoning": round(c.reasoning_tokens_per_call),
                "Assumed output": int(out_tokens_reasoning if row_meta[i][2].reasoning else out_tokens),
                "Summary tokens": round(c.summary_text_tokens),
                "Prompt-token factor": round(c.prompt_token_ratio, 3),
                "s / call": round(c.seconds_per_call, 2),
                "Calls ok": f"{c.n_link_calls + c.n_summary_calls}/{len(c.records)}",
                "Spent": c.spent_usd,
            }
            for i, c in cal_rows
        ]
    )
    st.dataframe(
        cal_df, hide_index=True, width="stretch",
        column_config={
            "Spent": st.column_config.NumberColumn("Spent", format="$%.4f"),
            "Prompt-token factor": st.column_config.NumberColumn(
                "Prompt-token factor", help="Provider-reported prompt tokens / our count (tokenizer + chat template)."
            ),
        },
    )
    for i, c in cal_rows:
        if c.n_errors:
            err = next(r.error for r in c.records if r.error)
            st.warning(f"**{rows[i]['dataset']} · {c.model}**: {c.n_errors} calls failed, e.g. {err[:300]}")

    with st.expander("See the real prompts and answers", icon=":material/chat:"):
        pick_cal = st.selectbox(
            "Run", options=range(len(cal_rows)), format_func=lambda k: cal_df.iloc[k]["Run"], key="cal_pick"
        )
        c = cal_rows[pick_cal][1]
        links = [r for r in c.records if r.kind == "link" and not r.error]
        sums = [r for r in c.records if r.kind == "summary" and not r.error]
        if links:
            k = st.slider("Prompt", 1, len(links), 1) if len(links) > 1 else 1
            r = links[k - 1]
            st.caption(f"Prompt sent ({r.prompt_tokens} tokens reported by the provider)")
            st.code(r.prompt, language=None, wrap_lines=True)
            st.caption(f"Answer ({r.completion_tokens} output tokens, {r.reasoning_tokens or 0} of them reasoning)")
            st.code(r.response or "(empty)", language=None, wrap_lines=True)
        if sums:
            st.caption("A real table summary")
            st.code(sums[0].response, language=None, wrap_lines=True)
    if st.button("Clear calibrations", icon=":material/delete:"):
        st.session_state.calibrations.clear()
        st.rerun()

# --------------------------------------------------------------------------- details

with st.expander("How was this estimated?", icon=":material/info:"):
    st.markdown(
        "- **Calls**: one per mention cell (repo datasets: every row x mention column; runs skip empty cells, so this is slightly high; "
        "SemTab: every annotated CEA cell) plus one table-summary call per table.\n"
        "- **Prompts**: rebuilt exactly as LionLinker builds them (template, 3-row table window, candidate "
        "formatting) and filled with real LamAPI candidates, then tokenized. Large datasets are sampled.\n"
        "- **Table summary and output tokens**: assumed until the run is calibrated with real calls (step 4); "
        "calibration also corrects for the provider's tokenizer and chat template.\n"
        "- **Forced GT candidates** are included when the option is on (default), as in real runs.\n"
        "- **Not included**: retries after API errors, OpenRouter prompt-caching discounts."
    )
    details = pd.DataFrame(
        [
            {
                "Dataset": p.dataset_name,
                "Prompt": SCHEMAS[p.schema],
                "Cells": p.n_tasks,
                "Tables": p.n_tables,
                "Prompts built": p.n_sampled,
                "Tokens / prompt (no summary)": round(p.mean_input),
                "Real candidates": f"{p.filled_rate:.0%}",
                "Fetched now": p.fetched,
                "Tokens / candidate": f"{p.tokens_per_candidate:.0f} ({p.tokens_per_candidate_source})",
                "Took": f"{p.elapsed_s:.1f}s",
            }
            for p in profiles
        ]
    )
    st.dataframe(details, hide_index=True, width="stretch")
    shown = set()
    for p in profiles:
        for w in p.warnings:
            if (p.dataset_name, w) not in shown:
                shown.add((p.dataset_name, w))
                st.warning(f"**{p.dataset_name}**: {w}")
    pick = st.selectbox(
        "Example prompt (offline; the summary shows as N.A. until calibrated)", options=range(len(profiles)),
        format_func=lambda i: f"{profiles[i].dataset_name} · {SCHEMAS[profiles[i].schema]}",
    )
    st.code(profiles[pick].example_prompt, language=None, wrap_lines=True)

stamp = datetime.now().strftime("%Y%m%d-%H%M")
d1, d2, _ = st.columns([1, 1, 3])
d1.download_button(
    "Download CSV", res.to_csv(index=False), file_name=f"cost_estimate_{stamp}.csv",
    mime="text/csv", icon=":material/download:",
)
d2.download_button(
    "Download JSON",
    json.dumps(
        {
            "created": stamp,
            "assumptions": {
                "limit": int(limit), "lamapi_limit": int(lamapi_limit), "output_tokens": int(out_tokens),
                "output_tokens_reasoning": int(out_tokens_reasoning), "summary_tokens": int(summary_tokens),
                "tokenizer": get_counter().name, "prices_source": prices.source, "prices_fetched_at": prices.fetched_at,
            },
            "runs": rows,
            "profiles": [{k: v for k, v in profile_to_dict(p).items() if k != "example_prompt"} for p in profiles],
            "calibrations": [c.to_dict(with_records=True) for _, c in cal_rows],
        },
        indent=2,
        default=str,
    ),
    file_name=f"cost_estimate_{stamp}.json",
    mime="application/json",
    icon=":material/download:",
)
