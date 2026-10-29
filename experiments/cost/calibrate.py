"""Calibrate an estimate with a handful of real OpenRouter calls.

For a few tables of the dataset this does exactly what a run does, with fully filled prompts:
1. asks the model for the table summary (LionLinker's summary prompt),
2. sends linking prompts with real LamAPI candidates and that real summary.

From the provider's ``usage`` it measures what the offline estimate can only assume:
real prompt tokens (provider tokenizer + chat template), output tokens incl. hidden reasoning,
and the summary length. Typical cost: a fraction of a cent.
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import pandas as pd
import requests

from experiments.cost.candidates import forced_ids_for, resolve_candidates
from experiments.cost.pricing import ModelPrice
from experiments.cost.prompts import build_prompt, summary_prompt
from experiments.cost.tasks import TaskSet
from experiments.cost.tokens import get_counter
from experiments.forcing import load_gt_ids
from experiments.runs.llm import usage_cost

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class CallRecord:
    kind: str  # "summary" | "link"
    table_id: str
    prompt: str
    response: str
    counted_prompt_tokens: int
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    seconds: float
    error: str | None = None


@dataclass
class Calibration:
    dataset_key: str
    schema: str
    model: str
    provider: str | None
    n_link_calls: int
    n_summary_calls: int
    n_errors: int
    prompt_token_ratio: float  # provider prompt tokens / our count
    output_tokens_per_call: float  # billed completion tokens (incl. reasoning), linking calls
    reasoning_tokens_per_call: float
    summary_text_tokens: float  # length of the real summary as pasted into prompts
    summary_call_output_tokens: float  # billed completion tokens of summary calls
    seconds_per_call: float
    spent_usd: float
    created_at: float
    records: list[CallRecord] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.n_link_calls > 0

    def to_dict(self, with_records: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not with_records:
            d.pop("records")
        return d


def openrouter_chat(model: str, content: str, provider: str | None, api_key: str, timeout: int = 180):
    """One user message, no extra parameters - the same request LionLinker makes."""
    body: dict[str, Any] = {"model": model, "messages": [{"role": "user", "content": content}]}
    if provider:
        body["provider"] = {"order": [provider], "allow_fallbacks": False}
    resp = requests.post(
        CHAT_URL,
        headers={"Authorization": f"Bearer {api_key}", "X-Title": "benchmarking-LLMs cost calibration"},
        json=body,
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(str(data["error"]))
    text = (data["choices"][0]["message"].get("content") or "").strip()
    return text, data.get("usage") or {}


def _call(kind: str, table_id: str, prompt: str, model: str, provider: str | None, key: str) -> CallRecord:
    counter = get_counter()
    started = time.time()
    try:
        text, usage = openrouter_chat(model, prompt, provider, key)
        details = usage.get("completion_tokens_details") or {}
        return CallRecord(
            kind, table_id, prompt, text, counter.count(prompt),
            usage.get("prompt_tokens"), usage.get("completion_tokens"), details.get("reasoning_tokens"),
            usage_cost(usage), time.time() - started,
        )
    except Exception as exc:
        return CallRecord(kind, table_id, prompt, "", counter.count(prompt), None, None, None, None,
                          time.time() - started, str(exc))


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


def calibrate(
    taskset: TaskSet,
    schema: str,
    price: ModelPrice,
    lamapi_clients: tuple[Any, Any],
    n_prompts: int = 20,
    max_tables: int = 5,
    seed: int = 13,
    api_key: str | None = None,
    workers: int = 8,
    progress: Callable[[float, str], None] | None = None,
    force_gt: bool = True,
) -> Calibration:
    from experiments.runs.endpoints import api_key_for

    key = api_key or api_key_for(None)
    if not key:
        raise RuntimeError("No OpenRouter API key: save one on the Endpoints page (or set OPENROUTER_API_KEY).")
    src = taskset.source
    rng = random.Random(seed)

    # Pick a few tables, then cells from them (round-robin so every table is represented).
    tables = list(dict.fromkeys(taskset.tasks["table_id"].tolist()))
    tables = rng.sample(tables, min(max_tables, len(tables)))
    by_table = {t: taskset.tasks[taskset.tasks["table_id"] == t] for t in tables}
    picked: list[tuple[str, int, int]] = []
    pools = {t: df.sample(frac=1, random_state=seed).itertuples(index=False) for t, df in by_table.items()}
    while len(picked) < n_prompts and pools:
        for t in list(pools):
            nxt = next(pools[t], None)
            if nxt is None:
                pools.pop(t)
                continue
            picked.append((t, nxt.row, nxt.col))
            if len(picked) >= n_prompts:
                break
    tables = list(dict.fromkeys(t for t, _, _ in picked))

    # 1) real table summaries
    if progress:
        progress(0.05, f"{src.name}: generating {len(tables)} table summaries")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        summary_recs = list(
            pool.map(
                lambda t: _call("summary", t, summary_prompt(taskset.load_table(t), seed), price.id, price.provider, key),
                tables,
            )
        )
    summaries = {r.table_id: r.response for r in summary_recs if not r.error}

    # 2) fully filled linking prompts (real candidates + real summary)
    if progress:
        progress(0.35, f"{src.name}: fetching candidates")
    gt_ids = load_gt_ids(src) if force_gt else {}
    lookups = {}
    for t, row, col in picked:
        df = taskset.load_table(t)
        if row < len(df) and col < len(df.columns):
            m = df.iloc[row, col]
            if not (pd.isna(m) or str(m).strip() == ""):
                lookups[(t, row, col)] = (m, forced_ids_for(gt_ids, t, row, col, force_gt))
    cands, _ = resolve_candidates(lookups.values(), src.target_kg, lamapi_clients, fetch_missing=True)

    jobs = []
    for t, row, col in picked:
        df = taskset.load_table(t)
        if row >= len(df) or col >= len(df.columns):
            continue
        k = lookups.get((t, row, col))
        c = cands.get(k) or [] if k else []
        jobs.append((t, build_prompt(schema, df, row, col, c, summaries.get(t))))
    if progress:
        progress(0.5, f"{src.name}: sending {len(jobs)} linking prompts")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        link_recs = list(pool.map(lambda j: _call("link", j[0], j[1], price.id, price.provider, key), jobs))

    counter = get_counter()
    ok_links = [r for r in link_recs if not r.error and r.prompt_tokens]
    ok_sums = [r for r in summary_recs if not r.error]
    all_recs = summary_recs + link_recs
    ratio = (
        sum(r.prompt_tokens for r in ok_links) / sum(r.counted_prompt_tokens for r in ok_links) if ok_links else 1.0
    )
    if progress:
        progress(1.0, f"{src.name}: calibrated")
    return Calibration(
        dataset_key=src.key,
        schema=schema,
        model=price.id,
        provider=price.provider,
        n_link_calls=len(ok_links),
        n_summary_calls=len(ok_sums),
        n_errors=sum(1 for r in all_recs if r.error),
        prompt_token_ratio=ratio,
        output_tokens_per_call=_mean([r.completion_tokens or 0 for r in ok_links]),
        reasoning_tokens_per_call=_mean([r.reasoning_tokens or 0 for r in ok_links]),
        summary_text_tokens=_mean([counter.count(" ".join(r.response.split())) for r in ok_sums]),
        summary_call_output_tokens=_mean([r.completion_tokens or 0 for r in ok_sums]),
        seconds_per_call=_mean([r.seconds for r in ok_links]),
        spent_usd=sum(
            r.cost_usd
            if r.cost_usd is not None
            else (r.prompt_tokens or 0) * price.input_per_token + (r.completion_tokens or 0) * price.output_per_token
            for r in all_recs
        ),
        created_at=time.time(),
        records=all_recs,
    )
