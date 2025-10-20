# services/riot_api_client/factory.py
from __future__ import annotations

from functools import lru_cache

from ..base import RiotAPI
from config.settings import settings

riot_api = RiotAPI()


@lru_cache(maxsize=1)
def get_riot_api() -> RiotAPI:
    """Global RiotHTTP client (dual-window rate limiter inside)."""
    return RiotAPI(
        calls=settings.rate_limit_calls,
        time_period=settings.rate_limit_period,
    )
