#!/usr/bin/env python3
"""Build the companies datasets."""

from __future__ import annotations

import argparse

from datasets.builders.build_utils import build_from_spec
from datasets.specs.registry import get_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build companies datasets.")
    parser.add_argument(
        "--include-crunchbase",
        action="store_true",
        help="Also build the Crunchbase GT variant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_from_spec(get_spec("cp"))
    if args.include_crunchbase:
        build_from_spec(get_spec("cp_crunchbase"))


if __name__ == "__main__":
    main()
