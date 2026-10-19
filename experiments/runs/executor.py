"""Run one LLM entity-linking benchmark with parallel requests.

Replaces ``LionLinker.run`` (which sends one request at a time and turns API errors into NIL)
while building byte-identical prompts through ``experiments.cost.prompts`` (LionLinker's own
PromptGenerator, checked by tests/test_cost_estimator.py).

Pipeline::

    cells --> [LamAPI workers x lamapi_concurrency] --> [LLM workers x max_concurrency] --> results.jsonl
                 candidates (+ forced GT ids)            table summary (once per table) + linking call

Every finished cell is appended to ``<run_dir>/results.jsonl``, so a cancelled, crashed or
re-queued run resumes where it stopped (cells that errored are retried). Live counters go to the
run store every couple of seconds; that is what the console's monitoring page reads.
"""

from __future__ import annotations

import asyncio
import logging
import json
import re
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from experiments.common.io import write_json
from experiments.common.logging import get_logger
from experiments.common.schemas import PRED_COLUMNS, build_key
from experiments.cost.candidates import _plain, make_lamapi_clients
from experiments.cost.prompts import build_prompt, summary_prompt
from experiments.cost.tasks import SPEC_TABLE_ID, TaskSet, load_taskset
from experiments.forcing import load_gt_ids
from experiments.runs.config import RunConfig
from experiments.runs.limiter import RateLimiter
from experiments.runs.llm import CallError, CallStats, FatalAPIError, OpenRouterClient
from experiments.runs.store import CANCELLED, CANCELLING, DONE, FAILED, RunStore

logger = get_logger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per request otherwise

ANSWER_RE = re.compile(r"ANSWER:\s*([^\s]+)", re.IGNORECASE)  # LionLinker's default pattern
TICK_S = 2.0
SENTINEL = object()


def extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text or "")
    return matches[-1] if matches else "No Identifier"


def cell_key(table_id: str, row: int, col: int) -> str:
    """Prediction key: ``row-col`` for the single-table repo datasets, ``table:row-col`` for SemTab."""
    return build_key(row, col) if table_id == SPEC_TABLE_ID else f"{table_id}:{build_key(row, col)}"


@dataclass
class Progress:
    n_tasks: int = 0
    n_resumed: int = 0  # cells already done by an earlier attempt of this run
    n_done: int = 0  # ok + empty + errors, including resumed ones
    n_ok: int = 0
    n_empty: int = 0  # empty mention: NIL without calling the model
    n_errors: int = 0
    n_tables: int = 0
    summaries_done: int = 0
    summary_errors: int = 0
    llm_calls: int = 0  # HTTP attempts (incl. retries and summaries)
    retries: int = 0
    rate_limited: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    lamapi_cached: int = 0
    lamapi_fetched: int = 0
    lamapi_failed: int = 0
    lamapi_seconds: float = 0.0  # summed over fetched lookups
    gt_cells: int = 0  # cells with a GT id
    gt_present: int = 0  # ... whose GT id ended up among the candidates
    phase: str = "starting"
    in_flight: int = 0
    cooling_down: bool = False
    max_concurrency: int = 0
    max_rpm: int = 0
    latency_p50_s: float | None = None
    latency_p90_s: float | None = None
    started_at: float = field(default_factory=time.time)
    session_started_at: float = field(default_factory=time.time)
    session_done_at_start: int = 0
    recent_errors: list[str] = field(default_factory=list)
    gt_missing_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class RunCancelled(Exception):
    pass


class RunExecutor:
    def __init__(
        self,
        config: RunConfig,
        store: RunStore | None = None,
        run_id: str | None = None,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
        api_key: str | None = None,
        lamapi_client=None,  # tests: a LamapiClient (e.g. offline on a fixture cache)
        llm_transport=None,  # tests: an httpx transport standing in for OpenRouter
        source=None,  # tests: a DatasetSource instead of resolving config.dataset
    ) -> None:
        self.cfg = config
        self._lamapi_client = lamapi_client
        self._llm_transport = llm_transport
        self.store = store
        self.run_id = run_id
        self.on_progress = on_progress
        self.api_key = api_key
        self.source = source or config.resolve_source()
        self.run_dir = config.run_dir(self.source)
        self.results_path = self.run_dir / "results.jsonl"
        self.summaries_path = self.run_dir / "summaries.json"
        self.p = Progress(max_concurrency=config.max_concurrency, max_rpm=config.max_rpm)
        self.stats = CallStats()
        self._latencies: deque[float] = deque(maxlen=500)
        self._errors: deque[str] = deque(maxlen=10)
        self._gt_missing: deque[str] = deque(maxlen=10)
        self._cancel = asyncio.Event()
        self._summaries: dict[str, str | None] = {}
        self._summary_tasks: dict[str, asyncio.Task] = {}
        self._lookups: dict[tuple, asyncio.Future] = {}
        self._fatal: Exception | None = None

    # ------------------------------------------------------------------ public

    def run(self) -> dict[str, Any]:
        return asyncio.run(self.arun())

    async def arun(self) -> dict[str, Any]:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        write_json(self.run_dir / "config.json", {**self.cfg.to_dict(), "dataset_key": self.source.key})
        taskset = load_taskset(self.source, self.cfg.limit)
        gt_ids = load_gt_ids(self.source)
        previous = self._load_previous(totals=self.p)
        self._summaries = self._load_summaries()

        tasks = list(taskset.tasks.itertuples(index=False))
        self.p.n_tasks = len(tasks)
        self.p.n_tables = taskset.n_tables
        self.p.summaries_done = sum(1 for v in self._summaries.values() if v is not None)
        self._account_previous(previous)
        todo = [t for t in tasks if previous.get(cell_key(t.table_id, t.row, t.col), {}).get("status") not in ("ok", "empty")]
        self.p.n_resumed = self.p.n_done
        self.p.session_done_at_start = self.p.n_done
        logger.info("%s: %d cells, %d already done, %d to go", self.cfg.label(), len(tasks), self.p.n_done, len(todo))

        status, error = DONE, None
        ticker = asyncio.create_task(self._ticker())
        try:
            if todo:
                await self._pipeline(taskset, todo, gt_ids)
            if self._cancel.is_set():
                status = CANCELLED  # (a fatal error raises above instead)
        except FatalAPIError as exc:
            status, error = FAILED, str(exc)
        except Exception as exc:  # noqa: BLE001 - reported to the store, then re-raised by the worker log
            logger.exception("run failed")
            status, error = FAILED, f"{type(exc).__name__}: {exc}"
        finally:
            ticker.cancel()
        self.p.phase = "writing outputs"
        summary = self._finalize(taskset, gt_ids, status, error)
        self.p.phase = status
        self._report()
        return summary

    # ------------------------------------------------------------------ pipeline

    async def _pipeline(self, taskset: TaskSet, todo: list, gt_ids: dict) -> None:
        cfg = self.cfg
        online = self._lamapi_client or make_lamapi_clients(cfg.lamapi_limit)[1]
        limiter = RateLimiter(cfg.max_concurrency, cfg.max_rpm)
        self._limiter = limiter
        llm = OpenRouterClient(
            cfg.model, limiter, cfg.provider, self.api_key, cfg.max_retries, stats=self.stats,
            transport=self._llm_transport,
        )
        in_q: asyncio.Queue = asyncio.Queue(maxsize=cfg.lamapi_concurrency * 4)
        ready_q: asyncio.Queue = asyncio.Queue(maxsize=cfg.max_concurrency * 4)
        self.p.phase = "running"

        with self.results_path.open("a", encoding="utf-8") as out:
            self._out = out

            async def feeder() -> None:
                for task in todo:
                    if self._cancel.is_set():
                        break
                    await in_q.put(task)
                for _ in range(cfg.lamapi_concurrency):
                    await in_q.put(SENTINEL)

            async def retrieval_worker() -> None:
                while (task := await in_q.get()) is not SENTINEL:
                    if self._cancel.is_set():
                        continue  # drain without fetching
                    try:
                        cell = await self._prepare(taskset, task, gt_ids, online)
                    except Exception as exc:  # noqa: BLE001
                        self._abort(exc)
                        continue
                    if cell is not None:
                        await ready_q.put(cell)

            async def llm_worker() -> None:
                # Workers never die on an error: they keep draining the queue so nothing upstream blocks.
                while (cell := await ready_q.get()) is not SENTINEL:
                    if self._cancel.is_set():
                        continue  # drain without calling
                    try:
                        await self._link(taskset, cell, llm)
                    except Exception as exc:  # noqa: BLE001
                        self._abort(exc)

            llm_workers = [asyncio.create_task(llm_worker()) for _ in range(cfg.max_concurrency)]
            try:
                await asyncio.gather(feeder(), *[retrieval_worker() for _ in range(cfg.lamapi_concurrency)])
                for _ in llm_workers:
                    await ready_q.put(SENTINEL)
                await asyncio.gather(*llm_workers)
            finally:
                for w in llm_workers:
                    w.cancel()
                await llm.aclose()
        if self._fatal is not None:
            raise self._fatal

    def _abort(self, exc: Exception) -> None:
        """Stop the run after an error that is not about a single cell (bad API key, no credits, bug)."""
        if self._fatal is None:
            self._fatal = exc
            if not isinstance(exc, FatalAPIError):
                logger.error("aborting run", exc_info=exc)
        self._cancel.set()

    async def _prepare(self, taskset: TaskSet, task, gt_ids: dict, client) -> dict | None:
        """Mention + candidates for one cell; records empty/failed cells directly."""
        df = taskset.load_table(task.table_id)
        key = cell_key(task.table_id, task.row, task.col)
        base = {"key": key, "table_id": task.table_id, "row": int(task.row), "col": int(task.col)}
        if task.row >= len(df) or task.col >= len(df.columns):
            self._record({**base, "status": "error", "error": "cell outside the table", "pred_id": "NIL"})
            return None
        mention = df.iloc[task.row, task.col]
        if pd.isna(mention) or str(mention).strip() == "":
            self._record({**base, "status": "empty", "pred_id": "NIL", "mention": None})
            return None
        mention = _plain(mention)
        gt = gt_ids.get((task.table_id, int(task.row), int(task.col)), [])
        forced = sorted(set((gt if self.cfg.force_gt else []) + list(self.cfg.force_ids))) or None
        base.update(mention=str(mention), gt_ids=gt, forced_ids=forced)
        try:
            candidates = await self._lookup(client, mention, forced)
        except Exception as exc:  # noqa: BLE001
            self.p.lamapi_failed += 1
            self._record({**base, "status": "error", "pred_id": "NIL", "error": f"LamAPI: {exc}"})
            return None
        ids = {str(c.get("id")) for c in candidates}
        gt_present = bool(set(gt) & ids) if gt else None
        if gt:
            self.p.gt_cells += 1
            self.p.gt_present += int(bool(gt_present))
            if not gt_present:
                self._gt_missing.append(f"{key} '{mention}': {' '.join(gt)}")
        return {**base, "candidates": candidates, "gt_in_candidates": gt_present, "df": df}

    async def _lookup(self, client, mention, forced: list[str] | None) -> list[dict]:
        target_kg = self.source.target_kg
        k = (repr(mention), tuple(forced or ()))
        fut = self._lookups.get(k)
        if fut is not None:  # same lookup already in flight: share it
            return await fut
        fut = asyncio.get_running_loop().create_future()
        self._lookups[k] = fut
        try:
            cached = client.is_cached(mention, ids=forced, target_kg=target_kg)
            started = time.monotonic()
            result = await asyncio.to_thread(client.fetch_entities, mention, forced, target_kg)
            if cached:
                self.p.lamapi_cached += 1
            else:
                self.p.lamapi_fetched += 1
                self.p.lamapi_seconds += time.monotonic() - started
            fut.set_result(result)
            return result
        except BaseException as exc:
            fut.set_exception(exc)
            fut.exception()  # mark retrieved for waiters that never come
            raise
        finally:
            self._lookups.pop(k, None)

    async def _summary(self, taskset: TaskSet, table_id: str, llm: OpenRouterClient) -> str | None:
        if table_id in self._summaries:
            return self._summaries[table_id]
        task = self._summary_tasks.get(table_id)
        if task is None:
            task = asyncio.create_task(self._make_summary(taskset, table_id, llm))
            self._summary_tasks[table_id] = task
        return await task

    async def _make_summary(self, taskset: TaskSet, table_id: str, llm: OpenRouterClient) -> str | None:
        prompt = summary_prompt(taskset.load_table(table_id), self.cfg.seed)
        try:
            res = await llm.chat(prompt)
        except CallError as exc:
            # The linking prompts then show "N.A." as summary, like a missing summary in LionLinker.
            self.p.summary_errors += 1
            self._errors.append(f"summary {table_id}: {exc}")
            self._summaries[table_id] = None
            return None
        self._account_call(res)
        text = res.text.strip() or None
        self._summaries[table_id] = text
        self.p.summaries_done += 1
        self._save_summaries()
        return text

    async def _link(self, taskset: TaskSet, cell: dict, llm: OpenRouterClient) -> None:
        df = cell.pop("df")
        candidates = cell.pop("candidates")
        summary = await self._summary(taskset, cell["table_id"], llm)
        prompt = build_prompt(self.cfg.schema, df, cell["row"], cell["col"], candidates, summary)
        cell["n_candidates"] = len(candidates)
        try:
            res = await llm.chat(prompt)
        except CallError as exc:
            self._record({**cell, "status": "error", "pred_id": "NIL", "error": str(exc)})
            return
        self._account_call(res)
        self._latencies.append(res.seconds)
        self._record(
            {
                **cell,
                "status": "ok",
                "pred_id": extract_answer(res.text),
                "answer": " ".join(res.text.split()),
                "prompt_tokens": res.prompt_tokens,
                "completion_tokens": res.completion_tokens,
                "reasoning_tokens": res.reasoning_tokens,
                "cost_usd": res.cost_usd,
                "latency_s": round(res.seconds, 3),
                "attempts": res.attempts,
                "provider": res.provider,
            },
            counted=True,
        )

    # ------------------------------------------------------------------ bookkeeping

    def _account_call(self, res) -> None:
        self.p.input_tokens += res.prompt_tokens
        self.p.output_tokens += res.completion_tokens
        self.p.reasoning_tokens += res.reasoning_tokens
        self.p.cost_usd += res.cost_usd or 0.0

    def _record(self, row: dict[str, Any], counted: bool = False) -> None:
        row["t"] = round(time.time(), 3)
        self._out.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")
        self._out.flush()
        self.p.n_done += 1
        status = row["status"]
        if status == "ok":
            self.p.n_ok += 1
        elif status == "empty":
            self.p.n_empty += 1
        else:
            self.p.n_errors += 1
            self._errors.append(f"{row['key']}: {row.get('error')}")

    def _load_previous(self, totals: Progress | None = None) -> dict[str, dict]:
        """Latest result per cell from results.jsonl; optionally add every attempt's spend to ``totals``."""
        latest: dict[str, dict] = {}
        if not self.results_path.exists():
            return latest
        with self.results_path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut short by a crash
                latest[row["key"]] = row
                if totals is not None:
                    totals.input_tokens += row.get("prompt_tokens") or 0
                    totals.output_tokens += row.get("completion_tokens") or 0
                    totals.reasoning_tokens += row.get("reasoning_tokens") or 0
                    totals.cost_usd += row.get("cost_usd") or 0.0
        return latest

    def _account_previous(self, previous: dict[str, dict]) -> None:
        for row in previous.values():
            if row.get("status") == "ok":
                self.p.n_done += 1
                self.p.n_ok += 1
                if row.get("gt_ids"):
                    self.p.gt_cells += 1
                    self.p.gt_present += int(bool(row.get("gt_in_candidates")))
            elif row.get("status") == "empty":
                self.p.n_done += 1
                self.p.n_empty += 1

    def _load_summaries(self) -> dict[str, str | None]:
        if self.summaries_path.exists():
            try:
                return {k: v for k, v in json.loads(self.summaries_path.read_text()).items() if v}
            except json.JSONDecodeError:
                pass
        return {}

    def _save_summaries(self) -> None:
        tmp = self.summaries_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: v for k, v in self._summaries.items() if v}, indent=1), encoding="utf-8")
        tmp.replace(self.summaries_path)

    def _snapshot(self) -> dict[str, Any]:
        self.p.llm_calls = self.stats.calls
        self.p.retries = self.stats.retries
        self.p.rate_limited = self.stats.rate_limited
        limiter = getattr(self, "_limiter", None)
        self.p.in_flight = limiter.in_flight if limiter else 0
        self.p.cooling_down = limiter.cooling_down if limiter else False
        if self._latencies:
            lat = sorted(self._latencies)
            self.p.latency_p50_s = statistics.median(lat)
            self.p.latency_p90_s = lat[min(len(lat) - 1, int(0.9 * len(lat)))]
        self.p.recent_errors = list(self._errors)
        self.p.gt_missing_examples = list(self._gt_missing)
        return self.p.to_dict()

    def _report(self) -> str | None:
        snap = self._snapshot()
        if self.on_progress:
            self.on_progress(snap)
        if self.store and self.run_id:
            return self.store.update_progress(self.run_id, snap, run_dir=str(self.run_dir))
        return None

    async def _ticker(self) -> None:
        while True:
            status = await asyncio.to_thread(self._report)
            if status in (CANCELLING, CANCELLED) and not self._cancel.is_set():
                logger.info("cancel requested")
                self.p.phase = "cancelling"
                self._cancel.set()
            await asyncio.sleep(TICK_S)

    # ------------------------------------------------------------------ outputs

    def _finalize(self, taskset: TaskSet, gt_ids: dict, status: str, error: str | None) -> dict[str, Any]:
        latest = self._load_previous()
        rows = []
        for t in taskset.tasks.itertuples(index=False):
            key = cell_key(t.table_id, t.row, t.col)
            r = latest.get(key)
            meta = {"status": "not_run"} if r is None else {
                k: r.get(k) for k in ("status", "mention", "answer", "error", "forced_ids", "gt_in_candidates",
                                      "n_candidates", "prompt_tokens", "completion_tokens", "cost_usd")
                if r.get(k) is not None
            }
            meta["model"] = self.cfg.model
            rows.append({
                "key": key,
                "pred_id": (r or {}).get("pred_id", "NIL"),
                "score": None,
                "runtime_ms": round(r["latency_s"] * 1000, 1) if r and r.get("latency_s") is not None else None,
                "meta_json": json.dumps(meta, ensure_ascii=True, default=str),
            })
        pred_df = pd.DataFrame(rows, columns=PRED_COLUMNS)
        pred_df.to_csv(self.run_dir / "predictions.csv", index=False)

        report = None
        if self.source.kind == "spec" and self.source.gt_path and Path(self.source.gt_path).exists():
            try:
                from experiments.evaluation.metrics import evaluate

                gt_df = pd.read_csv(self.source.gt_path)
                gt_df = gt_df[gt_df["key"].isin(pred_df["key"])]
                report = evaluate(pred_df, gt_df)
                write_json(self.run_dir / "report.json", report)
            except Exception as exc:  # noqa: BLE001 - evaluation is a bonus, never fail the run on it
                logger.warning("evaluation failed: %s", exc)

        manifest = {
            "run_id": self.run_id,
            "status": status,
            "error": error,
            "config": self.cfg.to_dict(),
            "dataset_key": self.source.key,
            "progress": self._snapshot(),
            "finished_at": time.time(),
            "report": report,
        }
        write_json(self.run_dir / "manifest.json", manifest)
        if self.store and self.run_id:
            self.store.finish(self.run_id, status, error)
        logger.info("%s: %s (%d ok, %d errors) -> %s", self.cfg.label(), status, self.p.n_ok, self.p.n_errors, self.run_dir)
        return manifest

