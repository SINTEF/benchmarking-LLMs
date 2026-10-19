"""Rebuild LionLinker prompts without instantiating LionLinker (which imports torch/ollama)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

# Copied from lion_linker.lion_linker.LionLinker.DEFAULT_ANSWER_FORMAT (whitespace-collapsed exactly
# as LionLinker does). tests/test_cost_estimator.py checks it stays in sync with the installed package.
DEFAULT_ANSWER_FORMAT = """
        Identify the correct identifier (ID) for the entity mention from the list of candidates above.

        Respond using the following format, and nothing else:

        ANSWER:{ID}

        Instructions:
        - Replace {ID} with the actual identifier (e.g., Q42, apple-234abc or Apple)
        - If none of the candidates is correct, respond with: ANSWER:NIL
        - Do not add any explanations, extra text, or formatting.
        - The output must be exactly one line and must start with 'ANSWER:'

        Examples:
        - ANSWER:Q42
        - ANSWER:apple-234abc
        - ANSWER:Apple
        - ANSWER:NIL
    """  # noqa: E501
ANSWER_FORMAT = " ".join(DEFAULT_ANSWER_FORMAT.split())

# Same mapping as experiments/runners/run_llm.py
PROMPT_SCHEMA_MAP = {"base": "base", "structured": "detailed", "fewshot": "few_shot"}

SUMMARY_INSTRUCTION = (
    "Provide a high-level summary of the table without getting into specific details. "
    "Reply only with the summary nothing else."
    "\nHere is a sample of the table data:\n"
)

# LionLinker defaults used by run_llm.py
CHUNK_SIZE = 64
TABLE_CTX_SIZE = 1


@lru_cache(maxsize=8)
def get_generator(schema: str):
    from lion_linker.prompt.generator import PromptGenerator

    return PromptGenerator(PROMPT_SCHEMA_MAP[schema], None, tablellama_format=False)


def table_context(df: pd.DataFrame, row: int) -> list[list[Any]]:
    """Header + row +/- TABLE_CTX_SIZE, clipped to the 64-row chunk LionLinker reads it in."""
    chunk_start = (row // CHUNK_SIZE) * CHUNK_SIZE
    chunk_end = min(chunk_start + CHUNK_SIZE, len(df))
    lo = max(chunk_start, row - TABLE_CTX_SIZE)
    hi = min(chunk_end, row + TABLE_CTX_SIZE + 1)
    view = df.iloc[lo:hi]
    return [view.columns.tolist()] + view.values.tolist()


def to_lionlinker_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Same conversion as run_llm._to_lionlinker_candidate."""
    return [
        {
            "id": c.get("id"),
            "name": c.get("label") or c.get("id"),
            "description": c.get("description"),
            "types": c.get("types") or [],
        }
        for c in candidates
    ]


def build_prompt(
    schema: str,
    df: pd.DataFrame,
    row: int,
    col: int,
    candidates: list[dict[str, Any]],
    table_summary: str | None = None,
) -> str:
    generator = get_generator(schema)
    column = df.columns[col]
    mention = df.iloc[row, col]
    return generator.generate_prompt(
        table=table_context(df, row),
        table_metadata=None,
        table_summary=table_summary,
        column_name=column,
        entity_mention=mention,
        candidates=to_lionlinker_candidates(candidates),
        compact=True,
        format_candidates=True,
        answer_format=ANSWER_FORMAT,
    )


def candidates_text(schema: str, candidates: list[dict[str, Any]]) -> str:
    """Just the [CANDIDATES] block, used to measure tokens per candidate."""
    if not candidates:
        return ""
    generator = get_generator(schema)
    full = generator.generate_prompt(
        table=[["x"]],
        table_metadata=None,
        table_summary=None,
        column_name="x",
        entity_mention="x",
        candidates=to_lionlinker_candidates(candidates),
        answer_format="",
    )
    empty = generator.generate_prompt(
        table=[["x"]],
        table_metadata=None,
        table_summary=None,
        column_name="x",
        entity_mention="x",
        candidates=[],
        answer_format="",
    )
    # The candidate block is the only difference between the two prompts.
    prefix = len(_common_prefix(full, empty))
    suffix_len = len(empty) - prefix
    return full[prefix : len(full) - suffix_len]


def summary_prompt(df: pd.DataFrame, seed: int = 13) -> str:
    """LionLinker.generate_table_summary input: 5 random rows of the first 64-row chunk."""
    chunk = df.head(CHUNK_SIZE)
    sample = chunk.sample(min(5, len(chunk)), random_state=seed) if len(chunk) else chunk
    return SUMMARY_INSTRUCTION + sample.to_string(index=False)


def _common_prefix(a: str, b: str) -> str:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return a[:i]
