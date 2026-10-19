"""Run the classic table linkers (Crocodile, Alligator) as monitored, resumable runs.

Same bookkeeping as LLM runs (``RunExecutor``: results.jsonl, resume, progress, P/R/F1): only
the pipeline differs. Each table is linked by one Crocodile/Alligator instance:

* rows: the table up to the last row with a target cell (``limit`` and SemTab targets respected);
* columns: the library's own column classifier, with every target column forced to NE so it is
  always linked (a target column classified as a literal would otherwise produce no prediction);
* forced candidates: the cell's GT ids (``correct_qids``, sent to LamAPI space-separated);
* prediction: the top candidate the library stores in MongoDB for the cell (NIL when none).

Both libraries need MongoDB (``MONGO_URI``) and tensorflow==2.18 (Python <= 3.12): they run in the
``linkers`` worker of docker-compose (Dockerfile.linkers).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict
from typing import Any

import pandas as pd

from experiments.common.logging import get_logger
from experiments.cost.candidates import _plain
from experiments.cost.tasks import TaskSet
from experiments.retrieval.lamapi_client import resolve_lamapi_endpoint, resolve_lamapi_token
from experiments.runs.config import METHODS
from experiments.runs.executor import RunExecutor, cell_key

logger = get_logger(__name__)

NE_CLASSES = {"PERSON", "OTHER", "ORGANIZATION", "LOCATION"}
DEFAULT_MONGO_URI = "mongodb://localhost:27017"


def lamapi_base(endpoint: str) -> str:
    """https://host/lookup/entity-retrieval -> https://host"""
    return endpoint.split("/lookup/", 1)[0].rstrip("/")


def column_types(df: pd.DataFrame, target_cols: set[int]) -> dict[str, Any]:
    """NE/LIT split from the library's column classifier, with every target column forced to NE."""
    ne: dict[str, str] = {}
    lit: dict[str, str] = {}
    try:
        from column_classifier import ColumnClassifier

        result = ColumnClassifier(model_type="fast").classify_multiple_tables([df.head(1024)])
        table = result[0].get("table_1", {}) if result else {}
    except Exception as exc:  # noqa: BLE001 - fall back to "targets are NE, the rest strings"
        logger.warning("column classification failed (%s); non-target columns treated as STRING", exc)
        table = {}
    for idx, col in enumerate(df.columns):
        label = (table.get(col) or {}).get("classification", "STRING")
        if idx in target_cols:
            ne[str(idx)] = label if label in NE_CLASSES else "OTHER"
        elif label in NE_CLASSES:
            ne[str(idx)] = label
        else:
            lit[str(idx)] = label if label != "UNKNOWN" else "STRING"
    return {"NE": ne, "LIT": lit, "IGNORED": []}


class LinkerExecutor(RunExecutor):
    """``RunExecutor`` with a per-table Crocodile/Alligator pipeline instead of LLM calls."""

    def _provider_preferences(self):  # no LLM provider involved
        return None

    async def _pipeline(self, taskset: TaskSet, todo: list, gt_ids: dict) -> None:
        method = self.cfg.method
        bow = (" · bag-of-words on" if self.cfg.bow else " · bag-of-words off") if method == "crocodile" else ""
        self.p.provider_route = f"{METHODS[method]} · {self.cfg.linker_workers} worker process(es) per table{bow}"
        self.p.max_concurrency = self.cfg.linker_workers
        by_table: OrderedDict[str, list] = OrderedDict()
        for t in todo:
            by_table.setdefault(t.table_id, []).append(t)
        self.p.tables_total = len(by_table)
        self.p.phase = "running"

        with self.results_path.open("a", encoding="utf-8") as out:
            self._out = out
            for i, (table_id, cells) in enumerate(by_table.items(), start=1):
                if self._cancel.is_set():
                    break
                self.p.phase = f"table {i}/{len(by_table)}: {table_id}"
                self.p.in_flight = len(cells)
                started = time.monotonic()
                try:
                    preds = await asyncio.to_thread(self._link_table, taskset, table_id, cells, gt_ids)
                except Exception as exc:  # noqa: BLE001 - one bad table must not stop the run
                    logger.exception("%s failed on table %s", method, table_id)
                    for t in cells:
                        self._record(self._base(t) | {"status": "error", "pred_id": "NIL",
                                                      "error": f"{type(exc).__name__}: {exc}"})
                    continue
                finally:
                    self.p.in_flight = 0
                seconds = time.monotonic() - started
                self._latencies.append(seconds / max(len(cells), 1))  # seconds per cell
                self.p.tables_done += 1
                for t in cells:
                    self._record_cell(t, preds, gt_ids, seconds / max(len(cells), 1))

    # ------------------------------------------------------------------ one table

    def _bow_endpoint(self, endpoint: str) -> str | None:
        if not self.cfg.bow:
            return None
        return os.getenv("ENTITY_BOW_ENDPOINT") or f"{lamapi_base(endpoint)}/entity/bow"

    def _base(self, t) -> dict[str, Any]:
        return {"key": cell_key(t.table_id, t.row, t.col), "table_id": t.table_id, "row": int(t.row), "col": int(t.col)}

    def _forced(self, t, gt_ids: dict) -> list[str]:
        gt = gt_ids.get((t.table_id, int(t.row), int(t.col)), []) if self.cfg.force_gt else []
        return sorted(set(gt) | set(self.cfg.force_ids))

    def _record_cell(self, t, preds: dict, gt_ids: dict, seconds: float) -> None:
        df_value = preds.get("__mentions__", {}).get((int(t.row), int(t.col)))
        base = self._base(t) | {"mention": df_value, "forced_ids": self._forced(t, gt_ids) or None}
        gt = gt_ids.get((t.table_id, int(t.row), int(t.col)), [])
        if df_value is None:
            self._record(base | {"status": "empty", "pred_id": "NIL"})
            return
        pred = preds.get((int(t.row), int(t.col)))
        candidates = pred["candidates"] if pred else []
        ids = {str(c.get("id")) for c in candidates}
        if gt:
            # Only the top candidates are stored by the library, so this is a lower bound.
            self.p.gt_cells += 1
            present = bool(set(gt) & ids)
            self.p.gt_present += int(present)
            if not present:
                self._gt_missing.append(f"{base['key']} '{df_value}': {' '.join(gt)} (not in the stored top candidates)")
        self._record(base | {
            "status": "ok",
            "pred_id": pred["id"] if pred and pred.get("id") else "NIL",
            "score": pred.get("score") if pred else None,
            "n_candidates": len(candidates),
            "gt_in_candidates": (bool(set(gt) & ids) if gt else None),
            "latency_s": round(seconds, 4),
        })

    def _link_table(self, taskset: TaskSet, table_id: str, cells: list, gt_ids: dict) -> dict:
        full = taskset.load_table(table_id)
        df = full.iloc[: max(int(t.row) for t in cells) + 1].copy()  # rows after the last target are not needed
        mentions = {}
        for t in cells:
            v = df.iloc[int(t.row), int(t.col)] if int(t.col) < len(df.columns) else None
            if v is not None and not pd.isna(v) and str(v).strip():
                mentions[(int(t.row), int(t.col))] = str(_plain(v))
        preds: dict[Any, Any] = {"__mentions__": mentions}
        if not mentions:
            return preds

        types = column_types(df, {int(t.col) for t in cells})
        forced = {f"{int(t.row)}-{int(t.col)}": self._forced(t, gt_ids) for t in cells}
        forced = {k: v for k, v in forced.items() if v}
        dataset_name = f"bench_{self.run_id or self.run_dir.name}"
        table_name = str(table_id)
        mongo_uri = os.getenv("MONGO_URI", DEFAULT_MONGO_URI)
        endpoint = resolve_lamapi_endpoint()
        token = resolve_lamapi_token()

        if self.cfg.method == "crocodile":
            from crocodile import Crocodile

            linker = Crocodile(
                input_csv=df, dataset_name=dataset_name, table_name=table_name, columns_type=types,
                max_workers=self.cfg.linker_workers, entity_retrieval_endpoint=endpoint,
                entity_retrieval_token=token, candidate_retrieval_limit=self.cfg.lamapi_limit,
                mongo_uri=mongo_uri, save_output_to_csv=False, return_dataframe=False,
                # Crocodile appends this string to the LamAPI URL as-is: space-separated ids
                correct_qids={k: " ".join(v) for k, v in forced.items()},
                # Crocodile does not read ENTITY_BOW_ENDPOINT itself: without this argument its
                # bow_similarity feature is 0.0 for every candidate.
                entity_bow_endpoint=self._bow_endpoint(endpoint),
            )
            db_name, collection, field = Crocodile._DB_NAME, Crocodile._INPUT_COLLECTION, "el_results"
        else:
            from alligator import Alligator

            base = lamapi_base(endpoint)
            linker = Alligator(
                input_csv=df, dataset_name=dataset_name, table_name=table_name, target_columns=types,
                target_rows=sorted({str(int(t.row)) for t in cells}), num_workers=self.cfg.linker_workers,
                entity_retrieval_endpoint=endpoint, entity_retrieval_token=token,
                object_retrieval_endpoint=os.getenv("OBJECT_RETRIEVAL_ENDPOINT", f"{base}/entity/objects"),
                literal_retrieval_endpoint=os.getenv("LITERAL_RETRIEVAL_ENDPOINT", f"{base}/entity/literals"),
                candidate_retrieval_limit=self.cfg.lamapi_limit, mongo_uri=mongo_uri,
                save_output=False, save_output_to_csv=False,
                correct_qids=forced,  # Alligator joins the list with spaces itself
            )
            db_name = linker.config.database.db_name
            collection = linker.config.database.input_collection
            field = "cea"
        linker.run()

        from pymongo import MongoClient

        client = MongoClient(mongo_uri)
        try:
            coll = client[db_name][collection]
            query = {"dataset_name": dataset_name, "table_name": table_name}
            for doc in coll.find(query, projection={"row_id": 1, field: 1}):
                row = int(doc["row_id"])
                for col, cands in (doc.get(field) or {}).items():
                    cands = cands or []
                    top = cands[0] if cands else {}
                    preds[(row, int(col))] = {"id": top.get("id"), "score": top.get("score"),
                                              "candidates": [{"id": c.get("id")} for c in cands]}
            coll.delete_many(query)  # keep MongoDB small; the libraries' candidate caches stay
        finally:
            client.close()
        return preds
