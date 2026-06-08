from __future__ import annotations

import asyncio
import logging
from typing import Any
from collections.abc import Callable, Mapping

from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

_logger = logging.getLogger(__name__)

# RECOVERY-SYSTEM: shared retry attempt ceiling. The per-call-site wait curves
# (DB ops here vs row inserts in the matchdata saver) are intentionally distinct
# and are NOT shared.
RETRY_MAX_ATTEMPTS = 8


# RECOVERY-SYSTEM: shared retry wrapper for sync persistence helpers.
@retry(
    stop=stop_after_attempt(RETRY_MAX_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    before_sleep=before_sleep_log(_logger, logging.WARNING),
    reraise=True,
)
async def run_sync_with_retry(
    *,
    logger: logging.Logger,
    component: str,
    op_name: str,
    func: Callable[..., Any],
    args: tuple[Any, ...] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> None:
    try:
        call_kwargs = dict(kwargs or {})
        await asyncio.to_thread(func, *args, **call_kwargs)
    except Exception as exc:
        logger.exception("%s %s failed: %s", component, op_name, exc)
        raise
