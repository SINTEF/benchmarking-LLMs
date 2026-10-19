"""Token counting. Uses tiktoken (o200k_base) when installed, else a character heuristic.

Every provider tokenizes differently, so treat counts as +/-10-15% for non-OpenAI models.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable


@dataclass(frozen=True)
class TokenCounter:
    name: str
    exact_for: str  # human-readable note shown in the UI
    _fn: Callable[[str], int]

    def count(self, text: str) -> int:
        return self._fn(text) if text else 0


CHARS_PER_TOKEN = 3.5  # conservative for tables full of IDs, numbers and punctuation


@lru_cache(maxsize=1)
def get_counter() -> TokenCounter:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return TokenCounter(
            "tiktoken o200k_base",
            "exact for recent OpenAI models (incl. gpt-oss); approximate for others",
            lambda t: len(enc.encode(t, disallowed_special=())),
        )
    except Exception:
        return TokenCounter(
            f"heuristic ({CHARS_PER_TOKEN} chars/token)",
            "rough approximation - `pip install tiktoken` for better counts",
            lambda t: math.ceil(len(t) / CHARS_PER_TOKEN),
        )
