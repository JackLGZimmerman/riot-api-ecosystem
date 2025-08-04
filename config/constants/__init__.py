
from .geography import (          # <-- only geography-related things
    Continent,
    Region,
    REGION_TO_CONTINENT,
    CONTINENT_TO_REGIONS,
)

from .parameters import (         # <-- ranked-queue / tier / division stuff
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


# 2) (optional) declare __all__ for wildcard imports
__all__ = [
    # geography
    "Continent", "Region", "REGION_TO_CONTINENT", "CONTINENT_TO_REGIONS",
    # ranked parameters
    "QUEUES", "ELITE_TIERS", "TIERS", "DIVISIONS",
    "DIVISION_MAPPING", "TIER_MAPPING", "ELITE_TIER_MAPPING", "QUEUE_TYPE_TO_QUEUE_CODE",
    # endpoints
    "ENDPOINTS",

]