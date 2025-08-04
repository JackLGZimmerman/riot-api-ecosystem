# config/constants/__init__.py
# -------------------------------------------------
# 1) Re-export the public names you want at package top level
from .geography import (          # <-- only geography-related things
    Continent,
    Region,
    REGION_TO_CONTINENT,
    CONTINENT_TO_REGIONS,
)

from .parameters import (         # <-- ranked-queue / tier / division stuff
    Queue,
    EliteTier,
    Tier,
    Division,
    QUEUES,
    ELITE_TIERS,
    TIERS,
    DIVISIONS,
    DIVISION_MAPPING,
    TIER_MAPPING,
    ELITE_TIER_MAPPING,
    QUEUE_TYPE_TO_QUEUE_CODE
)

from .endpoints import ENDPOINTS

from .http import (
    HTTP_OK,
    HTTP_BAD_REQUEST,
    HTTP_UNAUTHORIZED,
    HTTP_FORBIDDEN,
    HTTP_NOT_FOUND,
    HTTP_METHOD_NOT_ALLOWED,
    HTTP_UNSUPPORTED_MEDIA_TYPE,
    HTTP_RATE_LIMITED,
    HTTP_INTERNAL_SERVER_ERROR,
    HTTP_BAD_GATEWAY,
    HTTP_SERVICE_UNAVAILABLE,
    HTTP_GATEWAY_TIMEOUT,
    ERROR_MESSAGES,
    RETRYABLE_STATUS_CODES,
    CLIENT_ERROR_STATUS_CODES,
)

# 2) (optional) declare __all__ for wildcard imports
__all__ = [
    # geography
    "Continent", "Region", "REGION_TO_CONTINENT", "CONTINENT_TO_REGIONS",
    # ranked parameters
    "Queue", "EliteTier", "Tier", "Division", "QUEUES", "ELITE_TIERS", "TIERS", "DIVISIONS",
    "DIVISION_MAPPING", "TIER_MAPPING", "ELITE_TIER_MAPPING", "QUEUE_TYPE_TO_QUEUE_CODE",
    # endpoints
    "ENDPOINTS",
    # HTTP / error handling
    "HTTP_OK", "HTTP_BAD_REQUEST", "HTTP_UNAUTHORIZED", "HTTP_FORBIDDEN",
    "HTTP_NOT_FOUND", "HTTP_METHOD_NOT_ALLOWED", "HTTP_UNSUPPORTED_MEDIA_TYPE",
    "HTTP_RATE_LIMITED", "HTTP_INTERNAL_SERVER_ERROR", "HTTP_BAD_GATEWAY",
    "HTTP_SERVICE_UNAVAILABLE", "HTTP_GATEWAY_TIMEOUT",
    "ERROR_MESSAGES", "RETRYABLE_STATUS_CODES", "CLIENT_ERROR_STATUS_CODES",
]