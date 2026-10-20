"""Deleting runs: the history entry, and optionally the run's output folder."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from experiments.cost.tasks import REPO_ROOT
from experiments.runs.store import ACTIVE, Run, RunStore

CONTAINER_ROOT = "/app"  # docker-compose mounts the repo here; stored run_dir paths may use it
RUN_FILES = ("results.jsonl", "manifest.json", "config.json", "predictions.csv")


@dataclass
class DeleteReport:
    deleted: list[str] = field(default_factory=list)  # run ids removed from the history
    skipped_active: list[str] = field(default_factory=list)  # still queued/running: cancel first
    folders_removed: list[str] = field(default_factory=list)
    folders_kept: dict[str, str] = field(default_factory=dict)  # folder -> why it was kept


def local_run_dir(run: Run) -> Path | None:
    """The run's output folder on this machine (runs executed in Docker store /app/... paths)."""
    if not run.run_dir:
        return None
    path = Path(run.run_dir)
    if not path.exists() and run.run_dir.startswith(CONTAINER_ROOT + "/"):
        path = REPO_ROOT / Path(run.run_dir).relative_to(CONTAINER_ROOT)
    return path


def _safe_to_remove(path: Path, run: Run) -> str | None:
    """Why the folder must not be removed, or None. Only a run folder under its outputs dir qualifies."""
    if not path.exists():
        return "already gone"
    out = Path(run.config.get("output_dir") or "outputs")
    out = (out if out.is_absolute() else REPO_ROOT / out).resolve()
    resolved = path.resolve()
    if out not in resolved.parents:
        return f"not inside the outputs folder {out}"
    if not any((resolved / f).exists() for f in RUN_FILES):
        return "does not look like a run folder"
    return None


def _prune_empty_parents(path: Path, run: Run) -> None:
    """Remove <dataset>/<method> folders left empty, never the outputs folder itself."""
    out = Path(run.config.get("output_dir") or "outputs")
    out = (out if out.is_absolute() else REPO_ROOT / out).resolve()
    parent = path.resolve().parent
    while parent != out and out in parent.parents:
        try:
            parent.rmdir()  # only succeeds when empty
        except OSError:
            break
        parent = parent.parent


def delete_runs(store: RunStore, run_ids: list[str], outputs: bool = False) -> DeleteReport:
    """Delete finished runs from the history; with ``outputs`` also their output folders.

    Runs with identical settings share one folder (that is how a run resumes): such a folder is
    kept while another run in the history still uses it.
    """
    report = DeleteReport()
    runs = [r for r in (store.get(i) for i in run_ids) if r is not None]
    report.skipped_active = [r.id for r in runs if r.status in ACTIVE]
    targets = [r for r in runs if r.status not in ACTIVE]
    report.deleted = store.delete([r.id for r in targets])
    if not outputs:
        return report

    deleted = set(report.deleted)
    still_used = {
        str(local_run_dir(r).resolve()): r.id
        for r in store.list(100_000) if r.id not in deleted and local_run_dir(r) is not None
    }
    for run in targets:
        if run.id not in deleted:
            continue
        path = local_run_dir(run)
        if path is None or str(path) in report.folders_removed:
            continue
        key = str(path.resolve()) if path.exists() else str(path)
        if key in still_used:
            report.folders_kept[str(path)] = f"also used by run {still_used[key]}"
            continue
        if reason := _safe_to_remove(path, run):
            report.folders_kept[str(path)] = reason
            continue
        shutil.rmtree(path)
        report.folders_removed.append(str(path))
        _prune_empty_parents(path, run)
    return report
