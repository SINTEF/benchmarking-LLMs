"""Cache page: what is cached on disk, and how much of each dataset's LamAPI lookups it covers."""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from experiments.cost.candidates import make_lamapi_clients
from experiments.cost.tasks import REPO_ROOT, discover_sources, load_taskset
from experiments.runs.eta import lookup_plan
from experiments.runs.store import default_db_path


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


@st.cache_data(ttl=30, show_spinner="Scanning...")
def scan(path: str) -> dict:
    """File count, bytes, oldest/newest mtime under ``path`` (recursive, fast os.scandir walk)."""
    files = size = 0
    oldest, newest = None, None
    stack = [path]
    while stack:
        try:
            entries = list(os.scandir(stack.pop()))
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        for e in entries:
            if e.is_dir(follow_symlinks=False):
                stack.append(e.path)
            elif e.is_file(follow_symlinks=False):
                st_ = e.stat()
                files += 1
                size += st_.st_size
                oldest = st_.st_mtime if oldest is None else min(oldest, st_.st_mtime)
                newest = st_.st_mtime if newest is None else max(newest, st_.st_mtime)
    return {"files": files, "bytes": size, "oldest": oldest, "newest": newest}


def when(ts: float | None) -> str:
    return datetime.fromtimestamp(ts).strftime("%d %b %Y %H:%M") if ts else "—"


st.title("Cache")

lamapi_dir = Path(os.getenv("LAMAPI_CACHE_DIR") or REPO_ROOT / "data" / "lamapi_cache")
outputs_dir = REPO_ROOT / "outputs"
locations = [
    ("LamAPI candidates", lamapi_dir, "One JSON file per lookup (mention + limit + forced ids). Shared by estimates and runs."),
    ("OpenRouter prices & providers", REPO_ROOT / ".cache", "Model prices and provider lists, refreshed daily."),
    ("Run outputs", outputs_dir, "Per-run results.jsonl, predictions.csv, reports - and the run store."),
]

if st.button("Rescan", icon=":material/refresh:"):
    scan.clear()

cols = st.columns(len(locations))
rows = []
for col, (name, path, desc) in zip(cols, locations):
    s = scan(str(path))
    col.metric(name, human(s["bytes"]), help=desc)
    col.caption(f"{s['files']:,} files · updated {when(s['newest'])}")
    rows.append({"Cache": name, "Path": str(path), "Files": s["files"], "Size": human(s["bytes"]),
                 "Oldest": when(s["oldest"]), "Newest": when(s["newest"])})
db = default_db_path()
if db.exists():
    st.caption(f"Run store: `{db}` · {human(db.stat().st_size + sum(p.stat().st_size for p in db.parent.glob(db.name + '-*')))}")
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# --------------------------------------------------------------------------- run outputs

st.subheader("Run outputs by dataset")
out_rows = []
if outputs_dir.is_dir():
    for ds in sorted(p for p in outputs_dir.iterdir() if p.is_dir()):
        for method in sorted(p for p in ds.iterdir() if p.is_dir()):
            s = scan(str(method))
            n_runs = sum(1 for p in method.iterdir() if p.is_dir())
            out_rows.append({"Dataset": ds.name, "Method": method.name, "Run folders": n_runs,
                             "Size": human(s["bytes"]), "Updated": when(s["newest"])})
if out_rows:
    st.dataframe(pd.DataFrame(out_rows), hide_index=True, width="stretch")
else:
    st.caption("No run outputs yet.")

# --------------------------------------------------------------------------- coverage

st.subheader("LamAPI coverage per dataset")
st.caption(
    "Share of a dataset's lookups already in the LamAPI cache, with the same keys a run uses "
    "(GT ids forced, default candidate limit). Uncached lookups are fetched during the run and slow it down "
    "when LamAPI is the bottleneck. Running **Estimate cost** on a dataset fetches its sampled lookups."
)
c1, c2, c3 = st.columns([2, 1, 1])
sources = discover_sources(Path(os.environ["DATASETS_ROOT"]).expanduser() if os.getenv("DATASETS_ROOT") else None)
picked = c1.multiselect("Datasets", [s.key for s in sources], default=[s.key for s in sources if s.kind == "spec"])
lamapi_limit = c2.number_input("Candidates per mention", 1, 100, 10)
force_gt = c3.checkbox("GT ids forced", value=True)
if st.button("Check coverage", icon=":material/fact_check:", disabled=not picked):
    offline, _ = make_lamapi_clients(int(lamapi_limit))
    cov = []
    bar = st.progress(0.0)
    for i, key in enumerate(picked):
        src = next(s for s in sources if s.key == key)
        started = time.time()
        plan = lookup_plan(load_taskset(src), offline, force_gt)
        u = plan["unique_lookups"] or 1
        cov.append({"Dataset": src.name, "Cells": plan["cells"], "Unique lookups": plan["unique_lookups"],
                    "Cached": plan["cached"], "To fetch": plan["to_fetch"], "Coverage": plan["cached"] / u,
                    "Checked in": f"{time.time() - started:.1f}s"})
        bar.progress((i + 1) / len(picked))
    bar.empty()
    st.dataframe(
        pd.DataFrame(cov), hide_index=True, width="stretch",
        column_config={"Coverage": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="percent")},
    )
