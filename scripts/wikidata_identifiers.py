"""Generate identifier-only datasets from Wikidata for EL experiments."""

from __future__ import annotations

import csv
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List

import requests

WDQS_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "benchmarking-LLMs dataset generation (local script)"

REQUEST_SLEEP_SECONDS = 1.0
MAX_RETRIES = 5
TIMEOUT_SECONDS = 60


COMPANIES_QUERY_TEMPLATE = """
SELECT ?item (MIN(?crunchbase_id) AS ?crunchbase_id)
WHERE {{
  ?item wdt:P2088 ?crunchbase_id.
}}
GROUP BY ?item
ORDER BY ?item
LIMIT {limit}
OFFSET {offset}
"""

MOVIES_QUERY_TEMPLATE = """
SELECT ?item
WHERE {{
  ?item wdt:P31 wd:Q11424.
}}
ORDER BY ?item
LIMIT {limit}
OFFSET {offset}
"""


@dataclass(frozen=True)
class CompanyRow:
    wikidata_qid: str
    crunchbase_id: str


@dataclass(frozen=True)
class MovieRow:
    wikidata_qid: str


def _parse_qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _sparql_query(session: requests.Session, query: str) -> Dict:
    headers = {
        "Accept": "application/sparql+json",
        "User-Agent": USER_AGENT,
    }
    params = {
        "format": "json",
        "query": query,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(
                WDQS_ENDPOINT,
                headers=headers,
                params=params,
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException:
            response = None

        if response is None:
            sleep_seconds = REQUEST_SLEEP_SECONDS * (2**attempt)
            time.sleep(sleep_seconds)
            continue

        if response.status_code in {429, 500, 502, 503, 504}:
            sleep_seconds = REQUEST_SLEEP_SECONDS * (2**attempt)
            time.sleep(sleep_seconds)
            continue

        response.raise_for_status()
        time.sleep(REQUEST_SLEEP_SECONDS)
        return response.json()

    raise RuntimeError("WDQS query failed after retries.")


def _stream_paginated_rows(
    session: requests.Session,
    query_template: str,
    page_size: int,
) -> Iterator[Dict]:
    offset = 0
    while True:
        query = query_template.format(limit=page_size, offset=offset)
        data = _sparql_query(session, query)
        bindings = data.get("results", {}).get("bindings", [])
        if not bindings:
            break
        for row in bindings:
            yield row
        offset += page_size


def _reservoir_sample(
    rows: Iterable,
    sample_size: int,
    rng: random.Random,
) -> List:
    sample: List = []
    total = 0
    for row in rows:
        total += 1
        if total <= sample_size:
            sample.append(row)
            continue
        replace_index = rng.randint(1, total)
        if replace_index <= sample_size:
            sample[replace_index - 1] = row
    if total < sample_size:
        raise ValueError(
            f"Not enough rows to sample {sample_size}; only {total} available."
        )
    return sample


def _write_csv(path: str, fieldnames: List[str], rows: Iterable[Dict[str, str]]) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _stream_companies(session: requests.Session, page_size: int) -> Iterator[CompanyRow]:
    for row in _stream_paginated_rows(session, COMPANIES_QUERY_TEMPLATE, page_size):
        yield CompanyRow(
            wikidata_qid=_parse_qid(row["item"]["value"]),
            crunchbase_id=row["crunchbase_id"]["value"],
        )


def _stream_movies(session: requests.Session, page_size: int) -> Iterator[MovieRow]:
    for row in _stream_paginated_rows(session, MOVIES_QUERY_TEMPLATE, page_size):
        yield MovieRow(
            wikidata_qid=_parse_qid(row["item"]["value"]),
        )


def generate_companies_dataset(
    output_dir: str,
    seed: int = 42,
    sample_size: int = 1000,
    page_size: int = 1000,
) -> List[CompanyRow]:
    rng = random.Random(seed)
    with requests.Session() as session:
        sample = _reservoir_sample(_stream_companies(session, page_size), sample_size, rng)

    sample_sorted = sorted(sample, key=lambda row: row.wikidata_qid)
    identifiers_rows = []
    gt_wikidata_rows = []
    gt_crunchbase_rows = []

    for index, row in enumerate(sample_sorted, start=1):
        row_id = f"cp-{index:06d}"
        identifiers_rows.append(
            {
                "id": row_id,
                "wikidata_qid": row.wikidata_qid,
                "crunchbase_id": row.crunchbase_id,
            }
        )
        gt_wikidata_rows.append(
            {
                "id": row_id,
                "wikidata_qid": row.wikidata_qid,
            }
        )
        gt_crunchbase_rows.append(
            {
                "id": row_id,
                "crunchbase_id": row.crunchbase_id,
            }
        )

    _write_csv(
        os.path.join(output_dir, "companies_cp_identifiers.csv"),
        ["id", "wikidata_qid", "crunchbase_id"],
        identifiers_rows,
    )
    _write_csv(
        os.path.join(output_dir, "companies_cp_gt_wikidata.csv"),
        ["id", "wikidata_qid"],
        gt_wikidata_rows,
    )
    _write_csv(
        os.path.join(output_dir, "companies_cp_gt_crunchbase.csv"),
        ["id", "crunchbase_id"],
        gt_crunchbase_rows,
    )
    return sample_sorted


def generate_movies_dataset(
    output_dir: str,
    seed: int = 42,
    sample_size: int = 1000,
    page_size: int = 1000,
) -> List[MovieRow]:
    rng = random.Random(seed)
    with requests.Session() as session:
        sample = _reservoir_sample(_stream_movies(session, page_size), sample_size, rng)

    sample_sorted = sorted(sample, key=lambda row: row.wikidata_qid)
    identifiers_rows = []
    gt_wikidata_rows = []

    for index, row in enumerate(sample_sorted, start=1):
        row_id = f"mv-{index:06d}"
        identifiers_rows.append(
            {
                "id": row_id,
                "wikidata_qid": row.wikidata_qid,
            }
        )
        gt_wikidata_rows.append(
            {
                "id": row_id,
                "wikidata_qid": row.wikidata_qid,
            }
        )

    _write_csv(
        os.path.join(output_dir, "movies_mv_identifiers.csv"),
        ["id", "wikidata_qid"],
        identifiers_rows,
    )
    _write_csv(
        os.path.join(output_dir, "movies_mv_gt_wikidata.csv"),
        ["id", "wikidata_qid"],
        gt_wikidata_rows,
    )
    return sample_sorted


def generate_all(
    output_dir: str,
    companies_seed: int = 42,
    movies_seed: int = 42,
    sample_size: int = 1000,
    page_size: int = 1000,
) -> None:
    generate_companies_dataset(
        output_dir=output_dir,
        seed=companies_seed,
        sample_size=sample_size,
        page_size=page_size,
    )
    generate_movies_dataset(
        output_dir=output_dir,
        seed=movies_seed,
        sample_size=sample_size,
        page_size=page_size,
    )


if __name__ == "__main__":
    generate_all(output_dir="data")
