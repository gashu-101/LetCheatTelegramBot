"""Structured logging setup."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> logging.Logger:
    """Configure the root logger with a console handler (and optional rotating file)."""

    root = logging.getLogger()
    root.setLevel(level)

    # Wipe any default handlers so re-configuration is idempotent.
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_h = RotatingFileHandler(
            log_file, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
        )
        file_h.setFormatter(formatter)
        root.addHandler(file_h)

    # Quiet down chatty third-party loggers.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("telegram").setLevel(logging.INFO)
    logging.getLogger("telegram.ext").setLevel(logging.INFO)

    return logging.getLogger("exam_bot")
