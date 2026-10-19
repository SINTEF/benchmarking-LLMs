"""Candidate forcing helpers."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

from experiments.common.logging import get_logger
from experiments.common.schemas import build_key

logger = get_logger(__name__)


GT_ID_COLUMNS = ["gt_id", "entity", "wikidata_qid", "crunchbase_id", "qid", "id"]


def normalize_identifier(value) -> str:
    if value is None:
        return "NIL"
    text = str(value).strip()
    if not text:
        return "NIL"
    lowered = text.lower()
    if lowered in {"nil", "none", "nan"}:
        return "NIL"
    if "wikidata.org/entity/" in lowered:
        return text.rsplit("/", 1)[-1]
    if "wikidata.org/wiki/" in lowered:
        return text.rsplit("/", 1)[-1]
    if "crunchbase.com/organization/" in lowered:
        slug = text.split("crunchbase.com/organization/", 1)[-1]
        return slug.strip("/")
    return text


def _resolve_key(row: pd.Series) -> str:
    if "key" in row:
        return str(row["key"])
    if "row" in row and "col" in row:
        return build_key(int(row["row"]), int(row["col"]))
    if "idRow" in row and "idCol" in row:
        return build_key(int(row["idRow"]), int(row["idCol"]))
    raise ValueError("GT rows must include key or row/col or idRow/idCol columns.")


def _resolve_gt_id(row: pd.Series) -> str:
    for col in GT_ID_COLUMNS:
        if col in row:
            return normalize_identifier(row[col])
    raise ValueError("GT file is missing a known GT id column.")


def build_correct_qids_from_gt(gt_path: Path) -> dict[str, str | list[str]]:
    df = pd.read_csv(gt_path)
    mapping: dict[str, list[str]] = defaultdict(list)
    for _, row in df.iterrows():
        key = _resolve_key(row)
        gt_id = _resolve_gt_id(row)
        if gt_id == "NIL":
            continue
        mapping[key].append(gt_id)
    output: dict[str, str | list[str]] = {}
    for key, values in mapping.items():
        unique = sorted(set(values))
        output[key] = unique[0] if len(unique) == 1 else unique
    return output


def build_mention_to_qids_from_table(
    table_path: Path,
    gt_path: Path,
    mention_columns: Iterable[str],
    normalizer: Callable[[str], str] | None = None,
) -> dict[str, str | list[str]]:
    table_df = pd.read_csv(table_path)
    gt_df = pd.read_csv(gt_path)
    mention_columns = list(mention_columns)
    mapping: dict[str, list[str]] = defaultdict(list)

    for _, row in gt_df.iterrows():
        key = _resolve_key(row)
        parts = key.split("-")
        if len(parts) != 2:
            continue
        id_row = int(parts[0])
        id_col = int(parts[1])
        if id_row >= len(table_df) or id_col >= len(table_df.columns):
            continue
        col_name = table_df.columns[id_col]
        if col_name not in mention_columns:
            continue
        mention = table_df.iloc[id_row, id_col]
        if pd.isna(mention):
            continue
        mention_text = str(mention)
        if normalizer:
            mention_text = normalizer(mention_text)
        gt_id = _resolve_gt_id(row)
        if gt_id == "NIL":
            continue
        mapping[mention_text].append(gt_id)

    output: dict[str, str | list[str]] = {}
    for mention, values in mapping.items():
        unique = sorted(set(values))
        output[mention] = unique[0] if len(unique) == 1 else unique
    return output


def merge_forced_ids(
    gt_forcing: dict[str, str | list[str]],
    extra_ids: Iterable[str] | None,
    keys: Iterable[str] | None = None,
) -> dict[str, list[str]]:
    extra_ids = [item for item in (extra_ids or []) if item]
    if not extra_ids:
        return {key: [value] if isinstance(value, str) else list(value) for key, value in gt_forcing.items()}

    output: dict[str, list[str]] = {}
    if not gt_forcing and keys:
        for key in keys:
            output[key] = list(extra_ids)
        return output

    for key, value in gt_forcing.items():
        ids = [value] if isinstance(value, str) else list(value)
        merged = ids + [item for item in extra_ids if item not in ids]
        output[key] = merged
    return output


def assert_forced_present(candidates: list[dict], forced_ids: Iterable[str]) -> None:
    forced_set = {str(item) for item in forced_ids if item}
    if not forced_set:
        return
    seen = {str(candidate.get("id")) for candidate in candidates}
    missing = sorted(forced_set - seen)
    if missing:
        logger.warning("Forced ids missing from candidates: %s", ", ".join(missing))


def load_gt_ids(source) -> dict[tuple[str, int, int], list[str]]:
    """(table_id, row, col) -> every GT id of that cell, for a ``DatasetSource``.

    Rows are 0-based data rows, i.e. the coordinates used by ``TaskSet.tasks``. NIL cells are left
    out. A cell may have several GT ids (multiple GT rows, or SemTab URIs separated by spaces):
    all of them are kept so all of them can be forced into the candidate list.
    """
    from experiments.cost.tasks import SPEC_TABLE_ID

    mapping: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    if source.kind == "spec":
        if not source.gt_path or not Path(source.gt_path).exists():
            return {}
        df = pd.read_csv(source.gt_path)
        for _, row in df.iterrows():
            r, c = (int(x) for x in _resolve_key(row).split("-"))
            gt_id = _resolve_gt_id(row)
            if gt_id != "NIL":
                mapping[(SPEC_TABLE_ID, r, c)].append(gt_id)
    elif source.kind == "semtab":
        df = pd.read_csv(source.cea_path, header=None, dtype=str)
        for table_id, r, c, uris in df.iloc[:, :4].itertuples(index=False):
            try:
                key = (str(table_id), int(r) - 1, int(c))  # SemTab row 1 == first data row
            except (TypeError, ValueError):
                continue  # header line or malformed row
            for uri in str(uris).split() if isinstance(uris, str) else []:
                gt_id = normalize_identifier(uri)
                if gt_id != "NIL":
                    mapping[key].append(gt_id)
    return {key: sorted(set(ids)) for key, ids in mapping.items()}
