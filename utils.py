"""Misc helpers: PDF handling, hashing, mime sniffing."""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
from dataclasses import dataclass
from typing import List, Tuple

log = logging.getLogger(__name__)


def sha256_bytes(data: bytes) -> str:
    """Stable content hash used for de-duplicating identical uploads."""
    return hashlib.sha256(data).hexdigest()


def looks_like_pdf(data: bytes, file_name: str | None = None) -> bool:
    if data.startswith(b"%PDF-"):
        return True
    if file_name and file_name.lower().endswith(".pdf"):
        return True
    return False


def looks_like_image(mime: str | None, file_name: str | None) -> bool:
    if mime and mime.startswith("image/"):
        return True
    if not file_name:
        return False
    lower = file_name.lower()
    return lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".heic", ".bmp", ".gif"))


@dataclass
class PdfContent:
    """Extracted PDF payload. Prefer text if present, otherwise images."""

    text: str
    page_images: List[Tuple[bytes, str]]  # (image_bytes, mime)


async def extract_pdf(data: bytes, max_pages: int = 10) -> PdfContent:
    """Pull text and (when text is sparse) rendered page images from a PDF.

    Runs the blocking PyMuPDF work in a worker thread so the event loop stays
    responsive while a large PDF is being processed.
    """

    return await asyncio.to_thread(_extract_pdf_sync, data, max_pages)


def _extract_pdf_sync(data: bytes, max_pages: int) -> PdfContent:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:  # pragma: no cover - import guard
        raise RuntimeError(
            "PyMuPDF is required for PDF support (pip install pymupdf)"
        ) from e

    text_parts: List[str] = []
    page_images: List[Tuple[bytes, str]] = []

    with fitz.open(stream=data, filetype="pdf") as doc:
        page_count = min(len(doc), max_pages)
        for i in range(page_count):
            page = doc.load_page(i)
            text = page.get_text("text").strip()
            if text:
                text_parts.append(text)

        # If text extraction was thin, fall back to rendered images so Gemini
        # can read scanned pages directly.
        joined_text = "\n\n".join(text_parts).strip()
        if len(joined_text) < 40:
            for i in range(page_count):
                page = doc.load_page(i)
                # 2x zoom keeps file size reasonable while staying readable.
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                page_images.append((pix.tobytes("png"), "image/png"))

    return PdfContent(text=joined_text, page_images=page_images)


def truncate(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


# ---------------------------------------------------------------------------
# Markdown -> Telegram HTML
# ---------------------------------------------------------------------------
#
# Gemini frequently returns markdown (**bold**, *italic*, `code`, `*   ` bullets)
# even when told not to. Telegram won't render that as plain text — the
# asterisks/underscores show literally. We translate the small subset of
# markdown Gemini actually uses into Telegram-flavored HTML.

import html as _html
import re as _re

_CODE_FENCE_RE = _re.compile(r"```([\w+-]*)\n?(.*?)```", _re.DOTALL)
_INLINE_CODE_RE = _re.compile(r"`([^`\n]+)`")
_BOLD_RE = _re.compile(r"\*\*(.+?)\*\*", _re.DOTALL)
_BOLD_UNDER_RE = _re.compile(r"__(.+?)__", _re.DOTALL)
_STRIKE_RE = _re.compile(r"~~(.+?)~~", _re.DOTALL)
_ITALIC_STAR_RE = _re.compile(r"(?<![\*\w])\*(?!\s)([^\*\n]+?)(?<!\s)\*(?!\w)")
_ITALIC_UNDER_RE = _re.compile(r"(?<![_\w])_(?!\s)([^_\n]+?)(?<!\s)_(?!\w)")
_BULLET_RE = _re.compile(r"^([ \t]*)[\*\-]\s+", _re.MULTILINE)
_HEADING_RE = _re.compile(r"^\s*#{1,6}\s+(.+)$", _re.MULTILINE)


def markdown_to_telegram_html(text: str) -> str:
    """Convert the markdown subset Gemini emits into Telegram HTML.

    Telegram's HTML parse mode supports <b>, <i>, <s>, <u>, <code>, <pre>, <a>.
    Everything else must be HTML-escaped.
    """

    placeholders: dict[str, str] = {}
    counter = [0]

    def stash(html_value: str) -> str:
        key = f"\x00PH{counter[0]}\x00"
        counter[0] += 1
        placeholders[key] = html_value
        return key

    def on_fence(m: "_re.Match[str]") -> str:
        body = m.group(2)
        return stash("<pre><code>" + _html.escape(body) + "</code></pre>")

    def on_inline(m: "_re.Match[str]") -> str:
        return stash("<code>" + _html.escape(m.group(1)) + "</code>")

    # 1. Pull out code blocks first so their contents aren't reformatted.
    text = _CODE_FENCE_RE.sub(on_fence, text)
    text = _INLINE_CODE_RE.sub(on_inline, text)

    # 2. Escape everything else so stray <, >, & are safe.
    text = _html.escape(text)

    # 3. Bullet markers before italic so "*   " isn't read as italic.
    text = _BULLET_RE.sub(lambda m: m.group(1) + "• ", text)

    # 4. Headings -> bold lines.
    text = _HEADING_RE.sub(r"<b>\1</b>", text)

    # 5. Inline formatting.
    text = _BOLD_RE.sub(r"<b>\1</b>", text)
    text = _BOLD_UNDER_RE.sub(r"<b>\1</b>", text)
    text = _STRIKE_RE.sub(r"<s>\1</s>", text)
    text = _ITALIC_STAR_RE.sub(r"<i>\1</i>", text)
    text = _ITALIC_UNDER_RE.sub(r"<i>\1</i>", text)

    # 6. Restore code placeholders.
    for key, val in placeholders.items():
        text = text.replace(key, val)

    return text
