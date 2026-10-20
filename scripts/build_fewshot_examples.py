#!/usr/bin/env python3
"""Build the fixed few-shot examples used by the ``fewshot`` prompt schema.

LionLinker's few-shot template has an ``[EXAMPLES]`` slot that is filled from a file; without one
it says "N.A.". This script writes that file from *real* cases:

* the cells come from the train splits (``data/Movie/movies_train.csv``,
  ``data/Company/companies_train.csv``), which no benchmark dataset evaluates on, so the examples
  cannot leak answers;
* the answers are their Wikidata ground truth;
* the candidates are what LamAPI returns for the mention (same endpoint, limit and forced GT ids as
  a run), formatted exactly like the candidates in a real prompt.

Three examples, picked deterministically (seeded order, first cell that fits):

1. a film title shared by several candidates (>= 4 in total), the right one not ranked first;
2. a company name shared by several candidates (>= 4 in total), the right one not ranked first;
3. a NIL case: a mention whose correct entity is not among LamAPI's candidates, although some
   candidates carry the same name, so the right answer really is ANSWER:NIL.

Output: experiments/prompts/fewshot_examples.txt (+ .json with provenance). The file is committed
so every run uses the same examples; rerun this script only to change them.

    python scripts/build_fewshot_examples.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.common.env import load_dotenv  # noqa: E402
from experiments.cost.candidates import make_lamapi_clients  # noqa: E402
from experiments.cost.prompts import candidates_text, table_context, to_lionlinker_candidates  # noqa: E402
from experiments.forcing import normalize_identifier  # noqa: E402

OUT_TXT = REPO_ROOT / "experiments" / "prompts" / "fewshot_examples.txt"
OUT_JSON = OUT_TXT.with_suffix(".json")
LAMAPI_LIMIT = 10  # RunConfig default
SEED = 13
MAX_TRIES = 600
MIN_CANDIDATES = 4  # positive examples: enough candidates to make the choice non-trivial

SOURCES = {
    "film": ("data/Movie/movies_train.csv", "data/Movie/el_movies_gt_wikidata.csv", "movies_train", "Title"),
    "company": ("data/Company/companies_train.csv", "data/Company/el_companies_gt_wikidata.csv", "companies_train", "company"),
}


def load(kind: str) -> tuple[pd.DataFrame, dict[int, str], str]:
    table_path, gt_path, table_name, column = SOURCES[kind]
    df = pd.read_csv(REPO_ROOT / table_path)
    gt = pd.read_csv(REPO_ROOT / gt_path)
    gt = gt[(gt["tableName"] == table_name) & (gt["idCol"] == df.columns.get_loc(column))]
    ids = {int(r): normalize_identifier(e) for r, e in zip(gt["idRow"], gt["entity"])}
    return df, {r: q for r, q in ids.items() if q != "NIL"}, column


def same_name(candidates: list[dict], mention: str) -> int:
    m = mention.strip().lower()
    return sum(1 for c in candidates if str(c.get("label") or "").strip().lower() == m)


def find(kind: str, client, want_nil: bool, skip: set[tuple[str, int]]) -> dict:
    df, gt, column = load(kind)
    col = df.columns.get_loc(column)
    rows = sorted(gt)
    random.Random(SEED).shuffle(rows)
    for row in rows[:MAX_TRIES]:
        if (kind, row) in skip:
            continue
        mention = df.iloc[row, col]
        if pd.isna(mention) or len(str(mention).strip()) < 3:
            continue
        mention = str(mention)
        qid = gt[row]
        plain = client.fetch_entities(mention)  # what LamAPI retrieves by itself
        if same_name(plain, mention) < 2:
            continue  # not ambiguous: a poor example
        ids = [c["id"] for c in plain]
        if want_nil:
            if qid in ids:
                continue
            candidates, answer = plain, "NIL"  # the right entity is not offered: NIL is correct
        else:
            if qid not in ids or ids.index(qid) == 0 or len(ids) < MIN_CANDIDATES:
                continue  # want the right entity retrieved among several, but not simply the top one
            candidates = client.fetch_entities(mention, ids=[qid])  # exactly what a run (GT forced) sends
            answer = qid
        return {"kind": kind, "row": row, "column": column, "mention": mention, "gt": qid, "answer": answer,
                "candidates": candidates, "table": table_context(df, row)}
    raise SystemExit(f"No suitable {'NIL ' if want_nil else ''}{kind} example in the first {MAX_TRIES} rows.")


def render(i: int, title: str, ex: dict) -> str:
    table = "\n".join("|" + "|".join(map(str, r)) + "|" for r in ex["table"])
    cands = candidates_text("base", ex["candidates"])
    return (
        f"### Example {i}: {title}\n"
        f"- Table data around the mention:\n{table}\n"
        f"- Column name:\n{ex['column']}\n"
        f"- Entity mention:\n{ex['mention']}\n"
        f"- Candidate entities:\n{cands}\n"
        f"Answer:\nANSWER:{ex['answer']}\n"
    )


def main() -> None:
    load_dotenv(REPO_ROOT / ".env")
    _, client = make_lamapi_clients(LAMAPI_LIMIT)
    film = find("film", client, want_nil=False, skip=set())
    company = find("company", client, want_nil=False, skip=set())
    nil = find("film", client, want_nil=True, skip={("film", film["row"])})
    examples = [
        ("film title shared by several entities", film),
        ("company name shared by several entities", company),
        ("the correct entity is not among the candidates", nil),
    ]
    text = "\n---\n\n".join(render(i, title, ex) for i, (title, ex) in enumerate(examples, start=1))
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(text, encoding="utf-8")
    OUT_JSON.write_text(json.dumps(
        {
            "lamapi_endpoint": client.endpoint, "lamapi_limit": LAMAPI_LIMIT, "seed": SEED,
            "examples": [
                {k: v for k, v in ex.items() if k not in ("table", "candidates")}
                | {"title": title, "source": SOURCES[ex["kind"]][0],
                   "candidate_ids": [c["id"] for c in ex["candidates"]]}
                for title, ex in examples
            ],
        },
        indent=2, ensure_ascii=False,
    ), encoding="utf-8")
    print(text)
    print(f"\nwrote {OUT_TXT.relative_to(REPO_ROOT)} and {OUT_JSON.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
