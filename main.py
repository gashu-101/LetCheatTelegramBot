"""Entry point: load config, set up logging, run the bot."""

from __future__ import annotations

import sys

from bot import run
from config import Settings
from logger import setup_logging


def main() -> int:
    try:
        settings = Settings.from_env()
    except RuntimeError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 2

    log = setup_logging(level=settings.log_level, log_file=settings.log_file)
    log.info("Starting Telegram AI Exam Assistant bot")
    try:
        run(settings)
    except KeyboardInterrupt:
        log.info("Interrupted; shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
