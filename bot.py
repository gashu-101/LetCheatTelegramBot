"""Telegram application wiring: build the Application and register handlers."""

from __future__ import annotations

import logging

from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
)

from config import Settings
from gemini_client import GeminiClient
from handlers import (
    STATE_KEY,
    BotState,
    cmd_disable,
    cmd_enable,
    cmd_groups,
    cmd_ping,
    cmd_restart,
    cmd_stats,
    handle_document,
    handle_photo,
    handle_text,
)
from image_processor import DedupCache

log = logging.getLogger(__name__)


async def _on_error(update, context):  # noqa: ANN001
    """Catch-all so a single bad update never tears down the polling loop."""
    log.exception("handler error: %s", context.error)


def build_application(settings: Settings) -> Application:
    """Construct the python-telegram-bot Application with all handlers attached."""

    gemini = GeminiClient(
        api_key=settings.gemini_api_key,
        model=settings.model,
        max_retries=settings.max_retries,
        request_timeout_s=settings.request_timeout_s,
    )
    state = BotState(
        settings=settings,
        gemini=gemini,
        cache=DedupCache(max_size=settings.dedup_cache_size),
    )

    app = (
        ApplicationBuilder()
        .token(settings.bot_token)
        .concurrent_updates(True)
        .build()
    )
    app.bot_data[STATE_KEY] = state

    # Admin commands first so they short-circuit before the text catch-all.
    app.add_handler(CommandHandler("ping", cmd_ping))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("groups", cmd_groups))
    app.add_handler(CommandHandler("enable", cmd_enable))
    app.add_handler(CommandHandler("disable", cmd_disable))
    app.add_handler(CommandHandler("restart", cmd_restart))

    # Content handlers.
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )

    app.add_error_handler(_on_error)

    log.info(
        "Bot configured | model=%s | allowed_groups=%s | admin=%s",
        settings.model,
        sorted(settings.allowed_groups) or "ALL",
        settings.admin_id,
    )
    return app


def run(settings: Settings) -> None:
    """Blocking polling loop."""
    # Python 3.12+ removed the implicit event-loop creation that PTB 21's
    # `run_polling` relies on. Create and install one explicitly.
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    app = build_application(settings)
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True,
        close_loop=False,
    )
