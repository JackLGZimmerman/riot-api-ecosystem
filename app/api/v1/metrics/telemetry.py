from collections.abc import Callable
from typing import TypeAlias

from prometheus_client import Counter, Gauge

from app.core.config.constants import (
    Continent,
    Region,
)
from app.core.config.constants.generic import RETRYABLE

OnAcquire: TypeAlias = Callable[[Continent, "Region | None", float], None]

rate_limiter_location_rate = Gauge(
    "rate_limiter_location_rate",
    "Current rate limiter usage (calls/sec) per location",
    ["location"],
)

api_http_error_codes = Counter(
    "api_http_error_codes_total",
    "HTTP error codes returned by the rate limiter",
    ["http_error_code", "category"],
)


def classify_http_code(code: int) -> tuple[str, str]:
    if code in RETRYABLE:
        return str(code), "retryable"

    return str(code), "unexpected"


def export_http_error_code_counter(code: int) -> None:
    http_code, category = classify_http_code(code)

    api_http_error_codes.labels(
        http_error_code=http_code,
        category=category,
    ).inc()


def export_location_event(*, location: Continent | Region, rate: float) -> None:
    rate_limiter_location_rate.labels(
        location=location.value,
    ).set(rate)
