"""Estimate tokens and USD cost of LLM runs before launching them.

Three steps:

1. ``profile_prompts`` - offline, model-independent: builds fully filled prompts (real LamAPI
   candidates, LionLinker template and table window) for a sample of cells and counts tokens.
2. ``calibrate``       - optional, a few real calls per run (see calibrate.py): measures the real
   table summary, provider prompt tokens and output/reasoning tokens.
3. ``price_profile``   - applies one model/provider's prices, using calibration when available.

CLI:
    python -m experiments.cost.estimate --dataset mv --dataset HardTablesR2 \
        --schema base --model openai/gpt-oss-120b --calibrate 20
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from experiments.common.env import load_dotenv
from experiments.cost.calibrate import Calibration, calibrate
from experiments.cost.candidates import forced_ids_for, make_lamapi_clients, resolve_candidates
from experiments.cost.pricing import ModelPrice, load_endpoints, load_prices, manual_price, with_provider
from experiments.cost.prompts import build_prompt, candidates_text, summary_prompt
from experiments.cost.tasks import REPO_ROOT, DatasetSource, TaskSet, discover_sources, load_taskset
from experiments.cost.tokens import get_counter
from experiments.forcing import load_gt_ids

__all__ = ["make_lamapi_clients", "profile_prompts", "price_profile", "ProfileConfig", "PromptProfile"]

DEFAULT_TOKENS_PER_CANDIDATE = 30.0  # only for mentions whose candidates could not be retrieved
DEFAULT_OUTPUT_TOKENS = 10  # "ANSWER:Q12345" plus a little slack
DEFAULT_REASONING_OUTPUT_TOKENS = 400  # reasoning tokens are billed as output
DEFAULT_SUMMARY_TOKENS = 60  # LionLinker asks for a table summary once per input CSV

ProgressFn = Callable[[float, str], None]


@dataclass(frozen=True)
class ProfileConfig:
    schema: str = "base"
    sample_size: int = 1000
    seed: int = 13
    lamapi_limit: int = 10
    fetch_missing: bool = True  # fill prompts with real LamAPI candidates (free)
    force_gt: bool = True  # GT ids forced into the candidates, as runs do by default
    default_tokens_per_candidate: float = DEFAULT_TOKENS_PER_CANDIDATE


@dataclass
class PromptProfile:
    dataset_key: str
    dataset_name: str
    schema: str
    n_tasks: int
    n_tables: int
    total_available: int
    n_sampled: int
    exact: bool
    mean_input: float  # per prompt, with summary "N.A." (the summary is added at pricing time)
    std_input: float
    input_tokens: float
    input_ci95: float
    na_summary_tokens: int
    summary_calls: int
    summary_input_tokens: float
    cache_hits: int  # mentions whose candidates came from the cache
    fetched: int  # ... fetched from LamAPI now
    cache_misses: int  # ... unavailable -> estimated
    empty_mentions: int
    tokens_per_candidate: float
    tokens_per_candidate_source: str
    tokenizer: str
    example_prompt: str
    elapsed_s: float
    warnings: list[str] = field(default_factory=list)

    @property
    def filled_rate(self) -> float:
        total = self.cache_hits + self.fetched + self.cache_misses
        return (self.cache_hits + self.fetched) / total if total else 1.0


# --------------------------------------------------------------------------- profiling


def profile_prompts(
    taskset: TaskSet,
    cfg: ProfileConfig,
    lamapi_clients: tuple[Any, Any] = (None, None),
    progress: ProgressFn | None = None,
) -> PromptProfile:
    started = time.time()
    counter = get_counter()
    src = taskset.source
    warnings = list(taskset.warnings)

    n_total = taskset.n_tasks
    exact = n_total <= cfg.sample_size
    sample = taskset.tasks if exact else taskset.tasks.sample(cfg.sample_size, random_state=cfg.seed)

    # Collect the sampled cells, their mentions and the ids a run forces into their candidates.
    gt_ids = load_gt_ids(src) if cfg.force_gt else {}
    cells = []
    bad_index = 0
    for task in sample.itertuples(index=False):
        df = taskset.load_table(task.table_id)
        if task.row >= len(df) or task.col >= len(df.columns):
            bad_index += 1
            continue
        mention = df.iloc[task.row, task.col]
        empty = pd.isna(mention) or str(mention).strip() == ""
        forced = forced_ids_for(gt_ids, task.table_id, task.row, task.col, cfg.force_gt)
        cells.append((task, df, None if empty else (mention, forced)))

    # Real candidates for every distinct mention: cache first, then LamAPI (parallel).
    def cand_progress(done: int, total: int) -> None:
        if progress:
            progress(0.6 * done / total, f"{src.name}: fetching LamAPI candidates {done}/{total}")

    cand_map, cstats = resolve_candidates(
        [m for _, _, m in cells if m is not None], src.target_kg, lamapi_clients, cfg.fetch_missing,
        progress=cand_progress,
    )

    tokens: list[float] = []
    missing_idx: list[int] = []
    hit_cand_tokens = hit_cand_count = n_filled = 0
    empty_count = 0
    example = ""
    for i, (task, df, mention) in enumerate(cells):
        if progress and i % 50 == 0:
            progress(0.6 + 0.35 * i / max(len(cells), 1), f"{src.name}: building prompt {i + 1}/{len(cells)}")
        candidates: list[dict] = []
        if mention is None:
            empty_count += 1  # LionLinker drops NaN mentions before retrieval
        else:
            found = cand_map.get(mention)
            if found is None:
                missing_idx.append(len(tokens))
            else:
                candidates = found
                n_filled += 1
                if found:
                    hit_cand_tokens += counter.count(candidates_text(cfg.schema, found))
                    hit_cand_count += len(found)
        prompt = build_prompt(cfg.schema, df, task.row, task.col, candidates, None)
        if not example and candidates:
            example = prompt
        tokens.append(float(counter.count(prompt)))

    if n_filled >= 20 and hit_cand_count >= 20:  # enough real candidate lists to extrapolate from
        per_cand = hit_cand_tokens / hit_cand_count
        per_cand_src = f"measured on {hit_cand_count:,} real candidates"
        est_missing = per_cand * min(hit_cand_count / max(n_filled, 1), cfg.lamapi_limit)
    else:
        per_cand = cfg.default_tokens_per_candidate
        per_cand_src = "default assumption"
        est_missing = per_cand * cfg.lamapi_limit
    for idx in missing_idx:
        tokens[idx] += est_missing

    if not example and cells:
        task, df, _ = cells[0]
        example = build_prompt(cfg.schema, df, task.row, task.col, [], None)

    n = len(tokens)
    mean = sum(tokens) / n if n else 0.0
    std = math.sqrt(sum((t - mean) ** 2 for t in tokens) / (n - 1)) if n > 1 else 0.0
    if exact or n < 2:
        ci = 0.0
    else:
        fpc = math.sqrt((n_total - n) / (n_total - 1)) if n_total > 1 else 0.0
        ci = 1.96 * std / math.sqrt(n) * fpc * n_total

    table_ids = list(dict.fromkeys(sample["table_id"].tolist()))[:200]
    summary_lens = [counter.count(summary_prompt(taskset.load_table(t), cfg.seed)) for t in table_ids]
    summary_mean = sum(summary_lens) / len(summary_lens) if summary_lens else 0.0

    if bad_index:
        warnings.append(f"{bad_index} sampled cells point outside their table and were ignored.")
    if missing_idx:
        why = "LamAPI is not configured" if lamapi_clients[0] is None else (
            "fetching was switched off" if not cfg.fetch_missing else "LamAPI returned an error")
        warnings.append(
            f"{len(missing_idx)} of {n} sampled prompts have no real candidates ({why}); they are estimated at "
            f"{est_missing:.0f} candidate tokens each ({per_cand_src})."
        )
    if progress:
        progress(1.0, f"{src.name}: done")

    return PromptProfile(
        dataset_key=src.key,
        dataset_name=src.name,
        schema=cfg.schema,
        n_tasks=n_total,
        n_tables=taskset.n_tables,
        total_available=taskset.total_available,
        n_sampled=n,
        exact=exact,
        mean_input=mean,
        std_input=std,
        input_tokens=mean * n_total,
        input_ci95=ci,
        na_summary_tokens=counter.count("N.A."),
        summary_calls=taskset.n_tables,
        summary_input_tokens=summary_mean * taskset.n_tables,
        cache_hits=cstats["cached"],
        fetched=cstats["fetched"],
        cache_misses=cstats["failed"],
        empty_mentions=empty_count,
        tokens_per_candidate=per_cand,
        tokens_per_candidate_source=per_cand_src,
        tokenizer=counter.name,
        example_prompt=example,
        elapsed_s=time.time() - started,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- pricing


def default_output_tokens(price: ModelPrice | None) -> int:
    return DEFAULT_REASONING_OUTPUT_TOKENS if price and price.reasoning else DEFAULT_OUTPUT_TOKENS


def price_profile(
    profile: PromptProfile,
    price: ModelPrice,
    output_tokens_per_call: float,
    summary_tokens: float = DEFAULT_SUMMARY_TOKENS,
    calibration: Calibration | None = None,
) -> dict[str, Any]:
    """Cost of one run. A calibration (same dataset/schema/model) replaces the assumptions."""
    ratio = 1.0
    summary_call_out = summary_tokens + (output_tokens_per_call if price.reasoning else 0)
    if calibration is not None and calibration.ok:
        ratio = calibration.prompt_token_ratio
        output_tokens_per_call = calibration.output_tokens_per_call
        summary_tokens = calibration.summary_text_tokens
        summary_call_out = calibration.summary_call_output_tokens

    calls = profile.n_tasks + profile.summary_calls
    summary_delta = summary_tokens - profile.na_summary_tokens  # real summary replaces "N.A."
    link_input = (profile.input_tokens + profile.n_tasks * summary_delta) * ratio
    ci = profile.input_ci95 * ratio
    input_tokens = link_input + profile.summary_input_tokens * ratio
    output_tokens = profile.n_tasks * output_tokens_per_call + profile.summary_calls * summary_call_out

    def cost(inp: float) -> float:
        return inp * price.input_per_token + output_tokens * price.output_per_token + calls * price.per_request

    return {
        "dataset": profile.dataset_name,
        "schema": profile.schema,
        "model": price.id,
        "provider": price.provider or "OpenRouter list price",
        "llm_calls": calls,
        "input_tokens": round(input_tokens),
        "output_tokens": round(output_tokens),
        "input_usd_per_mtok": price.input_per_mtok,
        "output_usd_per_mtok": price.output_per_mtok,
        "cost_usd": cost(input_tokens),
        "cost_low_usd": cost(max(input_tokens - ci, 0)),
        "cost_high_usd": cost(input_tokens + ci),
        "output_tokens_per_call": output_tokens_per_call,
        "summary_tokens": summary_tokens,
        "prompt_token_ratio": ratio,
        "calibrated": calibration is not None and calibration.ok,
        "exact": profile.exact,
    }


def profile_to_dict(profile: PromptProfile) -> dict[str, Any]:
    d = asdict(profile)
    d["filled_rate"] = profile.filled_rate
    return d


# --------------------------------------------------------------------------- CLI


def _resolve_source(name: str, sources: list[DatasetSource]) -> DatasetSource:
    for s in sources:
        if name in (s.key, s.name):
            return s
    raise SystemExit(f"Unknown dataset '{name}'. Available: {', '.join(s.key for s in sources)}")


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", required=True, help="spec name or key (repeatable)")
    p.add_argument("--schema", action="append", choices=["base", "structured", "fewshot"])
    p.add_argument("--model", action="append", required=True, help="OpenRouter model id (repeatable)")
    p.add_argument("--datasets-root", default=None)
    p.add_argument("--limit", type=int, default=0, help="first N rows (spec) / cells (SemTab); 0 = all")
    p.add_argument("--sample-size", type=int, default=1000)
    p.add_argument("--lamapi-limit", type=int, default=10)
    p.add_argument("--no-fetch", action="store_true", help="use only cached LamAPI candidates")
    p.add_argument("--output-tokens", type=int, default=None, help="per call; default depends on model")
    p.add_argument("--summary-tokens", type=int, default=DEFAULT_SUMMARY_TOKENS)
    p.add_argument("--calibrate", type=int, default=0, metavar="N", help="send N real prompts per run to calibrate")
    p.add_argument("--input-price", type=float, default=None, help="USD per 1M input tokens (overrides OpenRouter)")
    p.add_argument("--output-price", type=float, default=None, help="USD per 1M output tokens (overrides OpenRouter)")
    p.add_argument("--reasoning", action="store_true", help="with manual prices: treat model as reasoning")
    p.add_argument(
        "--provider", default="cheapest",
        help="OpenRouter provider tag/name (e.g. groq, deepinfra/bf16), 'cheapest' (healthy) or 'list'",
    )
    p.add_argument("--json", type=str, default=None, help="write full results to this file")
    args = p.parse_args()

    root = Path(args.datasets_root).expanduser() if args.datasets_root else None
    sources = discover_sources(root)
    prices = load_prices()
    if prices.error:
        print(f"[prices] {prices.source}: {prices.error}")
    clients = make_lamapi_clients(args.lamapi_limit)

    rows, profiles, calibrations = [], [], []
    for ds in args.dataset:
        taskset = load_taskset(_resolve_source(ds, sources), args.limit)
        for schema in args.schema or ["base"]:
            cfg = ProfileConfig(
                schema=schema, sample_size=args.sample_size, lamapi_limit=args.lamapi_limit,
                fetch_missing=not args.no_fetch,
            )
            prof = profile_prompts(taskset, cfg, clients)
            profiles.append(prof)
            for w in prof.warnings:
                print(f"[{prof.dataset_name}] {w}")
            for model in args.model:
                if args.input_price is not None or args.output_price is not None:
                    price = manual_price(model, args.input_price or 0.0, args.output_price or 0.0, args.reasoning)
                else:
                    price = prices.find(model)
                if price is None:
                    print(f"[prices] no OpenRouter price for '{model}' - skipped")
                    continue
                out = args.output_tokens if args.output_tokens is not None else default_output_tokens(price)
                if args.input_price is None and args.output_price is None and args.provider != "list":
                    eps = load_endpoints(price.id)
                    chosen = (
                        eps.cheapest(prof.mean_input, out) if args.provider == "cheapest" else eps.by_tag(args.provider)
                    )
                    if chosen is None:
                        print(f"[prices] provider '{args.provider}' not found for {price.id}; using list price")
                    else:
                        price = with_provider(price, chosen)
                cal = None
                if args.calibrate:
                    cal = calibrate(taskset, schema, price, clients, n_prompts=args.calibrate, seed=cfg.seed)
                    calibrations.append(cal)
                    print(
                        f"[calibration] {prof.dataset_name}/{schema}/{price.id}@{price.provider}: "
                        f"{cal.output_tokens_per_call:.0f} output tok/call, prompt-token factor "
                        f"{cal.prompt_token_ratio:.3f}, summary {cal.summary_text_tokens:.0f} tok, "
                        f"spent ${cal.spent_usd:.4f}, errors {cal.n_errors}"
                    )
                rows.append(price_profile(prof, price, out, args.summary_tokens, cal))

    if rows:
        df = pd.DataFrame(rows)
        cols = ["dataset", "schema", "model", "provider", "calibrated", "llm_calls", "input_tokens",
                "output_tokens", "cost_usd", "cost_low_usd", "cost_high_usd"]
        with pd.option_context("display.width", 220, "display.float_format", "{:,.2f}".format):
            print(df[cols].to_string(index=False))
        print(f"\nTOTAL: ${df['cost_usd'].sum():,.2f}  (range ${df['cost_low_usd'].sum():,.2f} - ${df['cost_high_usd'].sum():,.2f})")
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "runs": rows,
                    "profiles": [profile_to_dict(x) for x in profiles],
                    "calibrations": [c.to_dict(with_records=True) for c in calibrations],
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
