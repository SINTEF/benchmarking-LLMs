"""Logging helpers shared across runners."""

from __future__ import annotations

import logging
import os


_DEFAULT_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with a single global handler."""
    root = logging.getLogger()
    if not root.handlers:
        level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(level=level, format=_DEFAULT_FORMAT)
    return logging.getLogger(name)
