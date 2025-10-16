from functools import lru_cache
from typing import TypeVar, Sequence, Mapping, Tuple, Final
from enum import StrEnum

# Define a generic enum type variable
E = TypeVar('E', bound=StrEnum)

class Queues(StrEnum):
    RANKED_SOLO_5x5 = "RANKED_SOLO_5x5"
    RANKED_FLEX_SR  = "RANKED_FLEX_SR"

QUEUE_TYPE_TO_QUEUE_CODE: Final[Mapping[Queues, int]] = {
    "RANKED_SOLO_5x5": 420,
    "RANKED_FLEX_SR": 440,
}

class EliteTiers(StrEnum):
    CHALLENGER = "CHALLENGER"
    GRANDMASTER  = "GRANDMASTER"
    MASTER  = "MASTER"

class Tiers(StrEnum):
    DIAMOND = "DIAMOND"
    EMERALD = "EMERALD"
    PLATINUM = "PLATINUM"
    GOLD = "GOLD"
    SILVER = "SILVER"
    BRONZE = "BRONZE"
    IRON = "IRON"

class Divisions(StrEnum):
    I = "I" 
    II = "II" 
    III = "III" 
    IV = "IV"


def cumulative_mapping(sequence: Sequence[E]) -> Mapping[E, Tuple[E, ...]]:
    return {value: sequence[: i + 1] for i, value in enumerate(sequence)}


@lru_cache(maxsize=None)
def cumulative_tier_mapping() -> Mapping[str, Tuple[str, ...]]:
    return cumulative_mapping(tuple(Tiers))


@lru_cache(maxsize=None)
def cumulative_elite_tier_mapping() -> Mapping[str, Tuple[str, ...]]:
    return cumulative_mapping(tuple(EliteTiers))


@lru_cache(maxsize=None)
def cumulative_division_mapping() -> Mapping[str, Tuple[str, ...]]:
    return cumulative_mapping(tuple(Divisions))