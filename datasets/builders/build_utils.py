"""Shared dataset build logic."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from datasets.specs.registry import DatasetSpec
from datasets.builders.common import (
    build_gt_rows,
    ensure_dir_for_file,
    filter_gt_to_table,
    write_metadata,
)


def build_from_spec(spec: DatasetSpec) -> dict[str, object]:
    table_path = Path(spec.source_table)
    gt_path = Path(spec.source_gt)
    if not table_path.exists():
        raise FileNotFoundError(f"Missing source table: {table_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing source GT: {gt_path}")

    table_df = pd.read_csv(table_path)
    if spec.table_columns:
        missing = [col for col in spec.table_columns if col not in table_df.columns]
        if missing:
            raise ValueError(f"Missing columns in {table_path}: {missing}")
        table_df = table_df[spec.table_columns]

    gt_df = pd.read_csv(gt_path)
    table_name = table_path.stem
    gt_df = filter_gt_to_table(gt_df, table_name)

    gt_rows = build_gt_rows(table_df, gt_df, spec.mention_columns, spec.target_kg, table_name)

    out_table = Path(spec.table_path)
    out_gt = Path(spec.gt_path)
    ensure_dir_for_file(out_table)
    ensure_dir_for_file(out_gt)

    table_df.to_csv(out_table, index=False)
    pd.DataFrame(gt_rows).to_csv(out_gt, index=False)

    metadata = {
        "name": spec.name,
        "description": spec.description,
        "source_table": spec.source_table,
        "source_gt": spec.source_gt,
        "table_path": spec.table_path,
        "gt_path": spec.gt_path,
        "mention_columns": spec.mention_columns,
        "target_kg": spec.target_kg,
        "seed": spec.seed,
        "table_rows": int(len(table_df)),
        "gt_rows": int(len(gt_rows)),
        "built_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    meta_path = out_table.parent / "metadata.json"
    write_metadata(meta_path, metadata)
    return metadata
