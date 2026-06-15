"""Telegram message and command handlers."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Set

from telegram import Update
from telegram.constants import ChatAction, ChatType, ParseMode
from telegram.ext import ContextTypes

from config import Settings, UNREADABLE_REPLY
from gemini_client import GeminiClient, GeminiError, GeminiPart
from image_processor import DedupCache, hash_payload
from utils import (
    extract_pdf,
    looks_like_image,
    looks_like_pdf,
    markdown_to_telegram_html,
    truncate,
)

log = logging.getLogger(__name__)


@dataclass
class Stats:
    started_at: float = field(default_factory=time.time)
    messages_seen: int = 0
    answered: int = 0
    failed: int = 0
    cache_hits: int = 0
    images: int = 0
    texts: int = 0
    pdfs: int = 0
    total_processing_s: float = 0.0


@dataclass
class BotState:
    settings: Settings
    gemini: GeminiClient
    cache: DedupCache
    stats: Stats = field(default_factory=Stats)
    semaphore: asyncio.Semaphore = field(init=False)
    # Runtime per-group enable/disable overlay. None = not toggled, True/False = override.
    runtime_groups: dict[int, bool] = field(default_factory=dict)
    disabled_groups: Set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.semaphore = asyncio.Semaphore(self.settings.max_concurrency)

    def is_chat_allowed(self, chat_id: int) -> bool:
        # Hard runtime disable wins.
        if chat_id in self.disabled_groups:
            return False
        # Runtime enable opens a group not in the env allowlist.
        if self.runtime_groups.get(chat_id) is True:
            return True
        return self.settings.is_group_allowed(chat_id)


# Stash the state on the application's `bot_data` so handlers can find it.
STATE_KEY = "bot_state"


def _state(context: ContextTypes.DEFAULT_TYPE) -> BotState:
    state = context.application.bot_data.get(STATE_KEY)
    if state is None:  # pragma: no cover - configuration error
        raise RuntimeError("BotState not attached to application")
    return state


def _is_admin(state: BotState, user_id: int | None) -> bool:
    return state.settings.admin_id is not None and user_id == state.settings.admin_id


# ---------------------------------------------------------------------------
# Core: send a request to Gemini with dedup + concurrency control.
# ---------------------------------------------------------------------------


async def _answer_with_cache(
    state: BotState, dedup_key: str, parts: list[GeminiPart]
) -> str:
    cached = await state.cache.get(dedup_key)
    if cached is not None:
        state.stats.cache_hits += 1
        return cached

    is_owner, fut = await state.cache.claim(dedup_key)
    if not is_owner:
        # Another coroutine is already calling Gemini for the same payload.
        return await fut

    try:
        async with state.semaphore:
            text = await state.gemini.answer(parts)
        await state.cache.complete(dedup_key, text)
        return text
    except BaseException as e:
        await state.cache.fail(dedup_key, e)
        raise


# ---------------------------------------------------------------------------
# Message handlers
# ---------------------------------------------------------------------------


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    if msg is None or chat is None or not msg.photo:
        return

    state = _state(context)
    state.stats.messages_seen += 1
    if not state.is_chat_allowed(chat.id):
        return

    # Highest-resolution variant is the last entry in the list.
    photo = msg.photo[-1]
    started = time.perf_counter()

    try:
        await context.bot.send_chat_action(chat.id, ChatAction.TYPING)

        file = await photo.get_file()
        data = bytes(await file.download_as_bytearray())

        dedup_key = hash_payload(b"photo", data)
        parts = [GeminiPart.of_image(data, mime="image/jpeg")]
        answer = await _answer_with_cache(state, dedup_key, parts)

        state.stats.images += 1
        await _finish(state, update, started, answer)

    except GeminiError as e:
        state.stats.failed += 1
        log.warning("Gemini error for photo (user=%s chat=%s): %s", user and user.id, chat.id, e)
        await _reply(msg, UNREADABLE_REPLY)
    except Exception as e:  # noqa: BLE001
        state.stats.failed += 1
        log.exception("Unhandled error processing photo: %s", e)


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    chat = update.effective_chat
    if msg is None or chat is None or not msg.text:
        return

    state = _state(context)
    state.stats.messages_seen += 1

    # Skip commands here — those are routed via CommandHandler.
    if msg.text.startswith("/"):
        return

    # In private chat, always respond. In groups, only if explicitly allowed.
    if chat.type != ChatType.PRIVATE and not state.is_chat_allowed(chat.id):
        return

    text = msg.text.strip()
    if not text:
        return

    started = time.perf_counter()
    try:
        await context.bot.send_chat_action(chat.id, ChatAction.TYPING)
        dedup_key = hash_payload(b"text", text)
        parts = [GeminiPart.of_text(text)]
        answer = await _answer_with_cache(state, dedup_key, parts)

        state.stats.texts += 1
        await _finish(state, update, started, answer)

    except GeminiError as e:
        state.stats.failed += 1
        log.warning("Gemini error for text: %s", e)
        await _reply(msg, UNREADABLE_REPLY)
    except Exception as e:  # noqa: BLE001
        state.stats.failed += 1
        log.exception("Unhandled error processing text: %s", e)


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.effective_message
    chat = update.effective_chat
    if msg is None or chat is None or msg.document is None:
        return

    state = _state(context)
    state.stats.messages_seen += 1
    if not state.is_chat_allowed(chat.id):
        return

    doc = msg.document
    file_name = doc.file_name or ""
    mime = doc.mime_type or ""

    started = time.perf_counter()
    try:
        await context.bot.send_chat_action(chat.id, ChatAction.TYPING)
        file = await doc.get_file()
        data = bytes(await file.download_as_bytearray())

        if looks_like_pdf(data, file_name):
            pdf = await extract_pdf(data, max_pages=state.settings.max_pdf_pages)
            parts: list[GeminiPart] = []
            if pdf.text:
                parts.append(GeminiPart.of_text(pdf.text))
            for img_bytes, img_mime in pdf.page_images:
                parts.append(GeminiPart.of_image(img_bytes, mime=img_mime))
            if not parts:
                await _reply(msg, UNREADABLE_REPLY)
                state.stats.failed += 1
                return

            dedup_key = hash_payload(b"pdf", data)
            answer = await _answer_with_cache(state, dedup_key, parts)
            state.stats.pdfs += 1
            await _finish(state, update, started, answer)
            return

        if looks_like_image(mime, file_name):
            dedup_key = hash_payload(b"doc-image", data)
            parts = [GeminiPart.of_image(data, mime=mime or "image/jpeg")]
            answer = await _answer_with_cache(state, dedup_key, parts)
            state.stats.images += 1
            await _finish(state, update, started, answer)
            return

        # Unknown file type — silently ignore in groups.
        log.debug("Ignoring unsupported document: %s (%s)", file_name, mime)

    except GeminiError as e:
        state.stats.failed += 1
        log.warning("Gemini error for document %s: %s", file_name, e)
        await _reply(msg, UNREADABLE_REPLY)
    except Exception as e:  # noqa: BLE001
        state.stats.failed += 1
        log.exception("Unhandled error processing document %s: %s", file_name, e)


async def _finish(state: BotState, update: Update, started: float, answer: str) -> None:
    elapsed = time.perf_counter() - started
    state.stats.answered += 1
    state.stats.total_processing_s += elapsed

    msg = update.effective_message
    chat = update.effective_chat
    user = update.effective_user
    log.info(
        "answered | user=%s(%s) | chat=%s | msg=%s | %.2fs | %s",
        getattr(user, "username", None) or "?",
        getattr(user, "id", "?"),
        getattr(chat, "id", "?"),
        getattr(msg, "message_id", "?"),
        elapsed,
        truncate(answer, 80),
    )
    if msg is not None:
        await _reply(msg, answer)


async def _reply(msg, text: str) -> None:
    """Reply to the original message.

    Gemini often emits markdown (**bold**, *italic*, `code`, "*  " bullets)
    even when told not to. We translate that small subset to Telegram HTML so
    it renders as intended; on parse failure we fall back to plain text.
    """
    html_text = markdown_to_telegram_html(text)
    try:
        await msg.reply_text(
            html_text,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
            do_quote=True,
        )
        return
    except Exception as e:  # noqa: BLE001 - malformed HTML, length, etc.
        log.warning("HTML reply_text failed (%s); retrying as plain text", e)

    try:
        await msg.reply_text(text, disable_web_page_preview=True, do_quote=True)
    except Exception as e:  # noqa: BLE001 - best-effort delivery
        log.warning("plain reply_text failed: %s", e)


# ---------------------------------------------------------------------------
# Admin commands
# ---------------------------------------------------------------------------


async def cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = _state(context)
    user = update.effective_user
    if not _is_admin(state, user.id if user else None):
        return
    await update.effective_message.reply_text("pong")


async def cmd_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = _state(context)
    user = update.effective_user
    if not _is_admin(state, user.id if user else None):
        return

    s = state.stats
    uptime_s = int(time.time() - s.started_at)
    avg = (s.total_processing_s / s.answered) if s.answered else 0.0
    body = (
        f"uptime: {uptime_s}s\n"
        f"model: {state.gemini.model}\n"
        f"messages_seen: {s.messages_seen}\n"
        f"answered: {s.answered}\n"
        f"failed: {s.failed}\n"
        f"cache_hits: {s.cache_hits}\n"
        f"images: {s.images} | texts: {s.texts} | pdfs: {s.pdfs}\n"
        f"avg_processing: {avg:.2f}s\n"
        f"cache_size: {len(state.cache)}\n"
        f"concurrency: {state.settings.max_concurrency}"
    )
    await update.effective_message.reply_text(body)


async def cmd_groups(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = _state(context)
    user = update.effective_user
    if not _is_admin(state, user.id if user else None):
        return

    allowed = sorted(state.settings.allowed_groups) or ["<all groups>"]
    runtime_enabled = sorted([g for g, v in state.runtime_groups.items() if v])
    disabled = sorted(state.disabled_groups)
    chat = update.effective_chat
    current = chat.id if chat else "?"
    body = (
        f"current chat: {current} (allowed={state.is_chat_allowed(chat.id) if chat else '?'})\n"
        f"env allowlist: {allowed}\n"
        f"runtime enabled: {runtime_enabled}\n"
        f"runtime disabled: {disabled}"
    )
    await update.effective_message.reply_text(body)


async def cmd_enable(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = _state(context)
    user = update.effective_user
    if not _is_admin(state, user.id if user else None):
        return
    chat = update.effective_chat
    if chat is None:
        return
    state.runtime_groups[chat.id] = True
    state.disabled_groups.discard(chat.id)
    await update.effective_message.reply_text(f"Enabled in chat {chat.id}")


async def cmd_disable(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = _state(context)
    user = update.effective_user
    if not _is_admin(state, user.id if user else None):
        return
    chat = update.effective_chat
    if chat is None:
        return
    state.disabled_groups.add(chat.id)
    state.runtime_groups[chat.id] = False
    await update.effective_message.reply_text(f"Disabled in chat {chat.id}")


async def cmd_restart(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    state = _state(context)
    user = update.effective_user
    if not _is_admin(state, user.id if user else None):
        return
    await update.effective_message.reply_text("Restarting…")
    log.warning("Admin %s requested restart", user.id if user else "?")
    # Exit non-zero so the supervisor (Docker, systemd, etc.) restarts us.
    import os
    os._exit(42)
