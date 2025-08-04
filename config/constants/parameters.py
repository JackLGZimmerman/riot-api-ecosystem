from typing import Final, List
from enum import StrEnum

class Queue(StrEnum):
    RANKED_SOLO_5x5 = "RANKED_SOLO_5x5"
    RANKED_FLEX_SR  = "RANKED_FLEX_SR"

class EliteTier(StrEnum):
    CHALLENGER  = "CHALLENGER"
    GRANDMASTER = "GRANDMASTER"
    MASTER      = "MASTER"    

class Tier(StrEnum):
    DIAMOND     = "DIAMOND"
    EMERALD     = "EMERALD"
    PLATINUM    = "PLATINUM"
    GOLD        = "GOLD"
    SILVER      = "SILVER"
    BRONZE      = "BRONZE"
    IRON        = "IRON"

class Division(StrEnum):
    I   = "I"
    II  = "II"
    III = "III"
    IV  = "IV"


QUEUES:   Final[list[Queue]]   = list(Queue)
TIERS:    Final[list[Tier]]    = list(Tier)
DIVISIONS:Final[list[Division]] = list(Division)
ELITE_TIERS: Final[List[EliteTier]] = list(EliteTier)

QUEUE_TYPE_TO_QUEUE_CODE: dict[str, int] = {
    "RANKED_SOLO_5x5": 420,
    "RANKED_FLEX_SR": 440,
}

DIVISION_MAPPING: Final[dict[Division, list[Division]]] = {
    div: DIVISIONS[: i + 1]
    for i, div in enumerate(DIVISIONS)
}

ELITE_TIER_MAPPING: Final[dict[EliteTier, List[EliteTier]]] = {
    tier: ELITE_TIERS[: i + 1]
    for i, tier in enumerate(ELITE_TIERS)
}

TIER_MAPPING: Final[dict[Tier, list[Tier]]] = {
    tier: TIERS[: i + 1]
    for i, tier in enumerate(TIERS)
}