#!/usr/bin/env python3
"""Build the SN dataset."""

from __future__ import annotations

from datasets.builders.build_utils import build_from_spec
from datasets.specs.registry import get_spec


def main() -> None:
    build_from_spec(get_spec("sn"))


if __name__ == "__main__":
    main()
