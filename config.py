"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set

from dotenv import load_dotenv

load_dotenv()


def _parse_int_set(raw: str | None) -> Set[int]:
    if not raw:
        return set()
    out: Set[int] = set()
    for piece in raw.replace(";", ",").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            out.add(int(piece))
        except ValueError:
            continue
    return out


def _parse_optional_int(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


@dataclass
class Settings:
    """Typed application settings."""

    bot_token: str
    gemini_api_key: str
    model: str = "gemini-2.5-flash"

    # Group allowlist. Empty set means "all groups allowed".
    allowed_groups: Set[int] = field(default_factory=set)

    # Telegram user id of the admin (optional).
    admin_id: int | None = None

    # Operational tunables.
    max_concurrency: int = 8
    request_timeout_s: float = 60.0
    max_retries: int = 4
    dedup_cache_size: int = 512
    max_pdf_pages: int = 10
    log_level: str = "INFO"
    log_file: Path | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        bot_token = os.getenv("BOT_TOKEN", "").strip()
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not bot_token:
            raise RuntimeError("BOT_TOKEN is required")
        if not gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required")

        log_file_raw = os.getenv("LOG_FILE", "").strip()
        return cls(
            bot_token=bot_token,
            gemini_api_key=gemini_api_key,
            model=os.getenv("MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash",
            allowed_groups=_parse_int_set(os.getenv("ALLOWED_GROUPS")),
            admin_id=_parse_optional_int(os.getenv("ADMIN_ID")),
            max_concurrency=int(os.getenv("MAX_CONCURRENCY", "8")),
            request_timeout_s=float(os.getenv("REQUEST_TIMEOUT_S", "60")),
            max_retries=int(os.getenv("MAX_RETRIES", "4")),
            dedup_cache_size=int(os.getenv("DEDUP_CACHE_SIZE", "512")),
            max_pdf_pages=int(os.getenv("MAX_PDF_PAGES", "10")),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_file=Path(log_file_raw) if log_file_raw else None,
        )

    def is_group_allowed(self, chat_id: int) -> bool:
        # Empty allowlist = open to all groups.
        if not self.allowed_groups:
            return True
        return chat_id in self.allowed_groups


# The single shared answer prompt. Used by every Gemini request.
ANSWER_PROMPT = """Return the answers only.

Do not explain your reasoning.
Do not repeat the questions.
Do not add introductions or conclusions.
If the image contains multiple-choice questions, return only the correct option letter and answer.
If there are numbered questions, preserve the numbering.

Example:

1. B
2. D
3. A
4. False
5. Photosynthesis

Output nothing except the answers."""


UNREADABLE_REPLY = (
    "Unable to read the image clearly. Please upload a higher-quality image."
)
