"""Async wrapper around the Google Gemini SDK with retries and rate-limit backoff."""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from config import ANSWER_PROMPT

log = logging.getLogger(__name__)


@dataclass
class GeminiPart:
    """A single multimodal input chunk: either text or an inline image."""

    text: str | None = None
    image_bytes: bytes | None = None
    mime_type: str | None = None

    @classmethod
    def of_text(cls, text: str) -> "GeminiPart":
        return cls(text=text)

    @classmethod
    def of_image(cls, data: bytes, mime: str = "image/jpeg") -> "GeminiPart":
        return cls(image_bytes=data, mime_type=mime)


class GeminiError(RuntimeError):
    """Raised when Gemini cannot produce a usable answer."""


class GeminiClient:
    """Thin async wrapper around `google-genai`.

    Lazily imports the SDK so module import doesn't fail when the package is
    missing during static analysis.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        *,
        max_retries: int = 4,
        request_timeout_s: float = 60.0,
    ) -> None:
        from google import genai  # type: ignore
        from google.genai import types as genai_types  # type: ignore

        self._genai = genai
        self._types = genai_types
        self._client = genai.Client(api_key=api_key)
        self._model = model
        self._max_retries = max_retries
        self._timeout = request_timeout_s

    @property
    def model(self) -> str:
        return self._model

    async def answer(self, parts: Sequence[GeminiPart]) -> str:
        """Run a single generation request and return the trimmed text."""

        contents = self._to_contents(parts)
        last_err: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._client.aio.models.generate_content(
                        model=self._model,
                        contents=contents,
                        config=self._types.GenerateContentConfig(
                            temperature=0.1,
                            # Keep the model laser-focused on the answer format.
                            system_instruction=ANSWER_PROMPT,
                        ),
                    ),
                    timeout=self._timeout,
                )
                text = (getattr(response, "text", None) or "").strip()
                if not text:
                    raise GeminiError("empty response from Gemini")
                return text

            except asyncio.TimeoutError as e:
                last_err = e
                log.warning("Gemini timed out (attempt %d/%d)", attempt, self._max_retries)
            except Exception as e:  # noqa: BLE001 - SDK raises a mix of types
                last_err = e
                msg = str(e).lower()
                is_rate_limited = (
                    "rate" in msg
                    or "quota" in msg
                    or "429" in msg
                    or "resource_exhausted" in msg
                )
                is_transient = is_rate_limited or any(
                    code in msg for code in ("500", "502", "503", "504", "unavailable")
                )
                log.warning(
                    "Gemini call failed (attempt %d/%d): %s",
                    attempt,
                    self._max_retries,
                    e,
                )
                if not is_transient and attempt > 1:
                    # Non-transient error after at least one retry: stop early.
                    break

            # Exponential backoff with jitter; longer wait if rate limited.
            backoff = min(2 ** attempt, 30) + random.uniform(0, 1.5)
            await asyncio.sleep(backoff)

        raise GeminiError(f"Gemini failed after {self._max_retries} attempts: {last_err}")

    # --- internals -----------------------------------------------------------

    def _to_contents(self, parts: Iterable[GeminiPart]) -> List[object]:
        """Convert our neutral GeminiPart list into google-genai Part objects."""

        out: List[object] = []
        Part = self._types.Part  # type: ignore[attr-defined]

        for p in parts:
            if p.image_bytes is not None:
                out.append(
                    Part.from_bytes(
                        data=p.image_bytes,
                        mime_type=p.mime_type or "image/jpeg",
                    )
                )
            elif p.text:
                out.append(Part.from_text(text=p.text))
        return out
