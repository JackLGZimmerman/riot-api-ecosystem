# services/riot_api_client/factory.py
from __future__ import annotations

from functools import lru_cache

from .base import RiotAPI
from config.settings import settings


@lru_cache(maxsize=1)
def get_riot_api() -> RiotAPI:
    """Global RiotHTTP client (dual-window rate limiter inside)."""
    return RiotAPI(
        api_key=settings.api_key,
        calls_per_second=settings.calls_per_second,
        calls_per_two_minutes=settings.calls_per_two_minutes,
    )
