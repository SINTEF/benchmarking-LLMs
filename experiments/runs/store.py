"""SQLite run store shared by the worker (writes) and the console (reads).

One row per run (config, status, live progress) plus periodic progress samples ("ticks") used for
throughput charts and the live ETA. WAL mode lets the console read while a worker writes.
"""

from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from experiments.cost.tasks import REPO_ROOT

DEFAULT_DB = REPO_ROOT / "outputs" / "runs.db"

QUEUED, RUNNING, CANCELLING, CANCELLED, DONE, FAILED, INTERRUPTED = (
    "queued", "running", "cancelling", "cancelled", "done", "failed", "interrupted",
)
ACTIVE = (QUEUED, RUNNING, CANCELLING)
FINISHED = (CANCELLED, DONE, FAILED, INTERRUPTED)
STALE_AFTER_S = 60  # a running run whose worker has not reported for this long is "interrupted"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    status TEXT NOT NULL,
    label TEXT,
    config TEXT NOT NULL,
    run_dir TEXT,
    worker TEXT,
    heartbeat REAL,
    started_at REAL,
    finished_at REAL,
    error TEXT,
    estimate TEXT,
    progress TEXT
);
CREATE INDEX IF NOT EXISTS runs_status ON runs(status, created_at);
CREATE TABLE IF NOT EXISTS ticks (
    run_id TEXT NOT NULL,
    t REAL NOT NULL,
    done INTEGER NOT NULL,
    errors INTEGER NOT NULL,
    llm_calls INTEGER NOT NULL,
    rate_limited INTEGER NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ticks_run ON ticks(run_id, t);
CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    host TEXT,
    pid INTEGER,
    started_at REAL,
    heartbeat REAL,
    current_run TEXT
);
"""
METHOD_SQL = "COALESCE(json_extract(config, '$.method'), 'llm')"  # runs from before methods were LLM runs


@dataclass
class Run:
    id: str
    created_at: float
    updated_at: float
    status: str
    label: str | None
    config: dict[str, Any]
    run_dir: str | None
    worker: str | None
    heartbeat: float | None
    started_at: float | None
    finished_at: float | None
    error: str | None
    estimate: dict[str, Any] = field(default_factory=dict)
    progress: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Run":
        d = dict(row)
        for key in ("config", "estimate", "progress"):
            d[key] = json.loads(d[key]) if d[key] else {}
        return cls(**d)

    @property
    def elapsed_s(self) -> float | None:
        if not self.started_at:
            return None
        return (self.finished_at or time.time()) - self.started_at


def default_db_path() -> Path:
    return Path(os.getenv("RUNS_DB") or DEFAULT_DB)


class RunStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(workers)")}
            if "methods" not in cols:  # stores created before workers were per method
                conn.execute("ALTER TABLE workers ADD COLUMN methods TEXT")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------ runs

    def submit(self, config: dict[str, Any], label: str | None = None, estimate: dict | None = None) -> str:
        run_id = uuid.uuid4().hex[:10]
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO runs (id, created_at, updated_at, status, label, config, estimate, progress) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, now, now, QUEUED, label, json.dumps(config), json.dumps(estimate or {}), "{}"),
            )
        return run_id

    def register_running(self, config: dict[str, Any], label: str | None, worker: str) -> str:
        """A run executed in-process (e.g. ``make run-llm``) rather than by the worker."""
        run_id = self.submit(config, label)
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, worker = ?, heartbeat = ?, started_at = ? WHERE id = ?",
                (RUNNING, worker, now, now, run_id),
            )
        return run_id

    def get(self, run_id: str) -> Run | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return Run.from_row(row) if row else None

    def list(self, limit: int = 200, statuses: tuple[str, ...] | None = None) -> list[Run]:
        query = "SELECT * FROM runs"
        args: list[Any] = []
        if statuses:
            query += f" WHERE status IN ({','.join('?' * len(statuses))})"
            args.extend(statuses)
        query += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._conn() as conn:
            return [Run.from_row(r) for r in conn.execute(query, args).fetchall()]

    def claim_next(self, worker_id: str, methods: tuple[str, ...] | None = None) -> Run | None:
        """Atomically move the oldest queued run (of one of ``methods``) to running for this worker."""
        now = time.time()
        query, args = "SELECT id FROM runs WHERE status = ?", [QUEUED]
        if methods:
            query += f" AND {METHOD_SQL} IN ({','.join('?' * len(methods))})"
            args.extend(methods)
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(query + " ORDER BY created_at LIMIT 1", args).fetchone()
            if row is None:
                conn.execute("COMMIT")
                return None
            conn.execute(
                "UPDATE runs SET status = ?, worker = ?, heartbeat = ?, started_at = ?, updated_at = ?, "
                "finished_at = NULL, error = NULL WHERE id = ?",
                (RUNNING, worker_id, now, now, now, row["id"]),
            )
            conn.execute("COMMIT")
        return self.get(row["id"])

    def update_progress(self, run_id: str, progress: dict[str, Any], run_dir: str | None = None) -> str:
        """Store live progress; returns the current status so the executor can notice a cancel."""
        now = time.time()
        with self._conn() as conn:
            if run_dir is not None:
                conn.execute("UPDATE runs SET run_dir = ? WHERE id = ?", (run_dir, run_id))
            conn.execute(
                "UPDATE runs SET progress = ?, heartbeat = ?, updated_at = ? WHERE id = ?",
                (json.dumps(progress), now, now, run_id),
            )
            conn.execute(
                "INSERT INTO ticks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id, now, progress.get("n_done", 0), progress.get("n_errors", 0),
                    progress.get("llm_calls", 0), progress.get("rate_limited", 0),
                    progress.get("input_tokens", 0), progress.get("output_tokens", 0),
                    progress.get("cost_usd", 0.0),
                ),
            )
            row = conn.execute("SELECT status FROM runs WHERE id = ?", (run_id,)).fetchone()
        return row["status"] if row else CANCELLED

    def finish(self, run_id: str, status: str, error: str | None = None) -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, error = ?, finished_at = ?, updated_at = ? WHERE id = ?",
                (status, error, now, now, run_id),
            )

    def cancel(self, run_id: str) -> None:
        """Queued runs are cancelled at once; running ones stop at the executor's next check."""
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status = CASE status WHEN ? THEN ? ELSE ? END, updated_at = ? "
                "WHERE id = ? AND status IN (?, ?)",
                (QUEUED, CANCELLED, CANCELLING, now, run_id, QUEUED, RUNNING),
            )
            conn.execute(
                "UPDATE runs SET finished_at = ? WHERE id = ? AND status = ?", (now, run_id, CANCELLED)
            )

    def requeue(self, run_id: str) -> None:
        """Queue a finished run again; the executor resumes from the cells already done."""
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status = ?, error = NULL, finished_at = NULL, updated_at = ? "
                f"WHERE id = ? AND status IN ({','.join('?' * len(FINISHED))})",
                (QUEUED, now, run_id, *FINISHED),
            )

    def delete(self, run_ids: list[str]) -> list[str]:
        """Remove finished runs and their progress samples from the store; returns the ids removed.

        Active runs (queued, running, cancelling) are never deleted: cancel them first.
        """
        if not run_ids:
            return []
        marks = ",".join("?" * len(run_ids))
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            removable = [
                r["id"] for r in conn.execute(
                    f"SELECT id FROM runs WHERE id IN ({marks}) AND status NOT IN ({','.join('?' * len(ACTIVE))})",
                    (*run_ids, *ACTIVE),
                )
            ]
            if removable:
                rm = ",".join("?" * len(removable))
                conn.execute(f"DELETE FROM ticks WHERE run_id IN ({rm})", removable)
                conn.execute(f"DELETE FROM runs WHERE id IN ({rm})", removable)
            conn.execute("COMMIT")
        return removable

    def mark_stale(self, max_age_s: float = STALE_AFTER_S) -> int:
        """Runs whose worker died (no heartbeat) become 'interrupted' so they can be resumed."""
        now = time.time()
        with self._conn() as conn:
            cur = conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, updated_at = ?, "
                "error = COALESCE(error, 'worker stopped responding') "
                "WHERE status IN (?, ?) AND COALESCE(heartbeat, 0) < ?",
                (INTERRUPTED, now, now, RUNNING, CANCELLING, now - max_age_s),
            )
            return cur.rowcount

    def ticks(self, run_id: str, since: float = 0.0) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM ticks WHERE run_id = ? AND t >= ? ORDER BY t", (run_id, since)
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ workers

    def worker_heartbeat(self, worker_id: str, current_run: str | None, methods: tuple[str, ...] = ("llm",)) -> None:
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO workers (id, host, pid, started_at, heartbeat, current_run, methods) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET heartbeat = excluded.heartbeat, current_run = excluded.current_run",
                (worker_id, socket.gethostname(), os.getpid(), now, now, current_run, ",".join(methods)),
            )

    def workers(self, alive_within_s: float = 30.0) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM workers WHERE heartbeat >= ? ORDER BY started_at", (time.time() - alive_within_s,)
            ).fetchall()
        return [dict(r) for r in rows]

    def remove_worker(self, worker_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM workers WHERE id = ?", (worker_id,))
