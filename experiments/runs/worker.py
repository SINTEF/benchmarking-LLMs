"""Background worker: picks queued runs from the run store and executes them one at a time.

    python -m experiments.runs worker

Runs survive console restarts and browser refreshes. Stopping the worker interrupts the current
run; queue it again and it resumes from the cells already done.
"""

from __future__ import annotations

import os
import signal
import socket
import threading
import time
import uuid

from experiments.common.logging import get_logger
from experiments.runs.config import RunConfig
from experiments.runs.executor import RunExecutor
from experiments.runs.store import FAILED, INTERRUPTED, RunStore

logger = get_logger(__name__)
POLL_S = 2.0


def run_worker(store: RunStore | None = None, once: bool = False) -> None:
    store = store or RunStore()
    worker_id = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:4]}"
    stop = threading.Event()
    current: dict[str, str | None] = {"run": None}

    def _stop(signum, _frame) -> None:
        logger.info("signal %s: stopping after the current poll (a running run is interrupted)", signum)
        stop.set()
        if current["run"] is None:
            raise SystemExit(0)
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)

    def heartbeat() -> None:
        while not stop.wait(5.0):
            store.worker_heartbeat(worker_id, current["run"])

    threading.Thread(target=heartbeat, daemon=True).start()
    store.worker_heartbeat(worker_id, None)
    logger.info("worker %s watching %s", worker_id, store.path)
    try:
        while not stop.is_set():
            store.mark_stale()
            run = store.claim_next(worker_id)
            if run is None:
                if once:
                    return
                time.sleep(POLL_S)
                continue
            current["run"] = run.id
            store.worker_heartbeat(worker_id, run.id)
            logger.info("run %s: %s", run.id, run.label)
            try:
                RunExecutor(RunConfig.from_dict(run.config), store, run.id).run()
            except KeyboardInterrupt:
                store.finish(run.id, INTERRUPTED, "worker stopped; resume to continue")
                raise
            except Exception as exc:  # config errors etc. - the executor reports its own failures
                logger.exception("run %s failed", run.id)
                store.finish(run.id, FAILED, f"{type(exc).__name__}: {exc}")
            finally:
                current["run"] = None
                store.worker_heartbeat(worker_id, None)
            if once:
                return
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        store.remove_worker(worker_id)
