from .geography import (
    Continents,
    Regions,
    REGION_TO_CONTINENT,
    CONTINENT_TO_REGIONS,
)

from .generic import JSON, JSONList

from .parameters import (
    Queues,
    EliteTiers,
    Tiers,
    Divisions,
    QUEUE_TYPE_TO_QUEUE_CODE,
)

from .endpoints import ENDPOINTS, URLTemplate


# 2) (optional) declare __all__ for wildcard imports
__all__ = [
    # geography
    "Continents",
    "Regions",
    "REGION_TO_CONTINENT",
    "CONTINENT_TO_REGIONS",
    # ranked parameters
    "Queues",
    "EliteTiers",
    "Tiers",
    "Divisions",
    "QUEUE_TYPE_TO_QUEUE_CODE",
    # endpoints
    "ENDPOINTS",
    "URLTemplate",
    # generics
    "JSON",
    "JSONList",
]
