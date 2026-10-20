"""Dataset sources and the list of LLM calls ("linking tasks") they imply.

Two kinds of sources are supported:

* ``spec``   - the repo's single-table datasets (``datasets/specs/*.json``).
               LionLinker makes one call per row x mention column (NaN cells included).
* ``semtab`` - SemTab-style folders under ``$DATASETS_ROOT``: ``tables/*.csv`` plus a
               CEA ground-truth file ``table_id,row,col,uri``. One call per annotated cell.
               Row indices follow SemTab: row 0 is the CSV header, row 1 the first data row.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import pandas as pd

from datasets.specs.registry import list_specs

REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_TABLE_ID = "__spec__"


@dataclass(frozen=True)
class DatasetSource:
    key: str  # unique id, e.g. "spec:mv" or "semtab:HardTablesR2"
    name: str
    kind: str  # "spec" | "semtab"
    description: str
    target_kg: str | None
    table_path: Path | None = None  # spec only
    mention_columns: tuple[str, ...] = ()  # spec only
    tables_dir: Path | None = None  # semtab only
    cea_path: Path | None = None  # semtab only
    gt_path: Path | None = None  # spec only (SemTab GT is cea_path)


@dataclass
class TaskSet:
    """All LLM calls for one dataset. ``tasks`` has columns table_id,row,col (0-based data rows)."""

    source: DatasetSource
    tasks: pd.DataFrame
    total_available: int  # before any limit was applied
    warnings: list[str] = field(default_factory=list)

    @property
    def n_tasks(self) -> int:
        return len(self.tasks)

    @property
    def n_tables(self) -> int:
        return int(self.tasks["table_id"].nunique()) if len(self.tasks) else 0

    def load_table(self, table_id: str) -> pd.DataFrame:
        if self.source.kind == "spec":
            return _read_table(str(self.source.table_path))
        return _read_table(str(self.source.tables_dir / f"{table_id}.csv"))


@lru_cache(maxsize=4096)
def _read_table(path: str) -> pd.DataFrame:
    # Default pandas parsing on purpose: this is what LionLinker does with its input CSV.
    try:
        return pd.read_csv(path)
    except pd.errors.ParserError:
        return pd.read_csv(path, engine="python", on_bad_lines="skip")


# --------------------------------------------------------------------------- discovery


def default_datasets_root() -> Path | None:
    value = os.getenv("DATASETS_ROOT")
    return Path(value).expanduser() if value else None


def _resolve_repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _find_cea_file(folder: Path) -> Path | None:
    """Prefer the Wikidata CEA GT (``cea*.csv``, table,row,col order) over 2019 DBpedia files."""
    gt_dir = folder / "gt"
    if not gt_dir.is_dir():
        return None
    preferred = sorted(gt_dir.glob("cea*.csv"))
    if preferred:
        return preferred[0]
    fallback = sorted(p for p in gt_dir.glob("*.csv") if "cea" in p.name.lower())
    return fallback[0] if fallback else None


def discover_sources(datasets_root: Path | None = None) -> list[DatasetSource]:
    sources: list[DatasetSource] = []
    for spec in list_specs().values():
        sources.append(
            DatasetSource(
                key=f"spec:{spec.name}",
                name=spec.name,
                kind="spec",
                description=spec.description,
                target_kg=spec.target_kg,
                table_path=_resolve_repo_path(spec.table_path),
                mention_columns=tuple(spec.mention_columns),
                gt_path=_resolve_repo_path(spec.gt_path) if spec.gt_path else None,
            )
        )
    root = datasets_root if datasets_root is not None else default_datasets_root()
    if root and root.is_dir():
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            cea = _find_cea_file(folder)
            if (folder / "tables").is_dir() and cea is not None:
                sources.append(
                    DatasetSource(
                        key=f"semtab:{folder.name}",
                        name=folder.name,
                        kind="semtab",
                        description=f"SemTab CEA ({cea.relative_to(folder)})",
                        target_kg="wikidata",
                        tables_dir=folder / "tables",
                        cea_path=cea,
                    )
                )
    return sources


def datasets_root_problem(root: str | Path | None = None) -> str | None:
    """Why no SemTab dataset is found under the datasets folder, or None when some are."""
    value = root if root not in (None, "") else os.getenv("DATASETS_ROOT")
    if not value:
        return "DATASETS_ROOT is not set in .env, so only the repo datasets are available."
    path = Path(value).expanduser()
    if not path.is_dir():
        return f"Datasets folder {path} does not exist."
    if any(s.kind == "semtab" for s in discover_sources(path)):
        return None
    nested = [p.name for p in path.parent.iterdir() if p.is_dir() and p != path
              and (p / "tables").is_dir() and _find_cea_file(p)] if path.parent.is_dir() else []
    hint = f" The parent folder {path.parent} has some ({', '.join(nested[:4])}...) - point DATASETS_ROOT there." if nested else ""
    return (f"No SemTab dataset in {path}: each dataset needs <name>/tables/*.csv and <name>/gt/cea*.csv.{hint}")


# --------------------------------------------------------------------------- task lists


def load_taskset(source: DatasetSource, limit: int = 0) -> TaskSet:
    """Build the task list. ``limit`` keeps the first N rows (spec) or first N cells (SemTab)."""
    if source.kind == "spec":
        return _spec_taskset(source, limit)
    if source.kind == "semtab":
        return _semtab_taskset(source, limit)
    raise ValueError(f"Unknown source kind: {source.kind}")


def _spec_taskset(source: DatasetSource, limit: int) -> TaskSet:
    df = _read_table(str(source.table_path))
    total_rows = len(df)
    n_rows = min(limit, total_rows) if limit else total_rows
    col_idx = []
    for name in source.mention_columns:
        if name in df.columns:
            col_idx.append(df.columns.get_loc(name))
        else:
            matches = [i for i, c in enumerate(df.columns) if c.lower() == name.lower()]
            if len(matches) != 1:
                raise ValueError(f"{source.name}: mention column '{name}' not found")
            col_idx.append(matches[0])
    rows = [(SPEC_TABLE_ID, r, c) for r in range(n_rows) for c in col_idx]
    tasks = pd.DataFrame(rows, columns=["table_id", "row", "col"])
    return TaskSet(source, tasks, total_available=total_rows * len(col_idx))


def _semtab_taskset(source: DatasetSource, limit: int) -> TaskSet:
    raw = pd.read_csv(source.cea_path, header=None, dtype=str, usecols=[0, 1, 2])
    raw.columns = ["table_id", "row", "col"]
    raw = raw.dropna()
    raw["row"] = pd.to_numeric(raw["row"], errors="coerce")
    raw["col"] = pd.to_numeric(raw["col"], errors="coerce")
    raw = raw.dropna().astype({"row": int, "col": int})
    raw = raw.drop_duplicates(["table_id", "row", "col"])
    warnings: list[str] = []
    header_rows = int((raw["row"] < 1).sum())
    if header_rows:
        warnings.append(f"{header_rows} GT cells point at the header row and were skipped.")
        raw = raw[raw["row"] >= 1]
    raw["row"] = raw["row"] - 1  # SemTab row 1 == first data row
    total = len(raw)
    if limit:
        raw = raw.head(limit)
    return TaskSet(source, raw.reset_index(drop=True), total_available=total, warnings=warnings)
