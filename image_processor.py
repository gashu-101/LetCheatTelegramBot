"""Image dedup cache + lightweight preprocessing."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from utils import sha256_bytes

log = logging.getLogger(__name__)


@dataclass
class CachedAnswer:
    text: str
    created_at: float


class DedupCache:
    """LRU cache of `content_hash -> answer text`.

    Two flavors of de-dup live here:
    1. Cached completed answers — return immediately on a second hit.
    2. In-flight tasks — a second concurrent identical request waits on the
       first and shares its result, so we never burn a duplicate Gemini call.
    """

    def __init__(self, max_size: int = 512) -> None:
        self._max = max_size
        self._answers: "OrderedDict[str, CachedAnswer]" = OrderedDict()
        self._in_flight: dict[str, asyncio.Future[str]] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            entry = self._answers.get(key)
            if entry is None:
                return None
            self._answers.move_to_end(key)
            return entry.text

    async def put(self, key: str, text: str) -> None:
        async with self._lock:
            self._answers[key] = CachedAnswer(text=text, created_at=time.time())
            self._answers.move_to_end(key)
            while len(self._answers) > self._max:
                self._answers.popitem(last=False)

    async def claim(self, key: str) -> tuple[bool, asyncio.Future[str]]:
        """Try to claim exclusive responsibility for filling `key`.

        Returns (is_owner, future). The owner must eventually call
        `complete()` / `fail()` for that key.
        """
        async with self._lock:
            fut = self._in_flight.get(key)
            if fut is not None:
                return False, fut
            new_fut: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            self._in_flight[key] = new_fut
            return True, new_fut

    async def complete(self, key: str, text: str) -> None:
        await self.put(key, text)
        async with self._lock:
            fut = self._in_flight.pop(key, None)
        if fut is not None and not fut.done():
            fut.set_result(text)

    async def fail(self, key: str, err: BaseException) -> None:
        async with self._lock:
            fut = self._in_flight.pop(key, None)
        if fut is not None and not fut.done():
            fut.set_exception(err)

    def __len__(self) -> int:
        return len(self._answers)


def hash_payload(*chunks: bytes | str) -> str:
    """Hash an ordered list of payload pieces into a single dedup key."""
    h_parts = []
    for c in chunks:
        if isinstance(c, str):
            h_parts.append(c.encode("utf-8"))
        else:
            h_parts.append(c)
    return sha256_bytes(b"\x1f".join(h_parts))


__all__ = ["DedupCache", "hash_payload", "sha256_bytes"]
