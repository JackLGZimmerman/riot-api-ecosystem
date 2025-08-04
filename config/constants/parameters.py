from functools import lru_cache
from typing import Tuple, Mapping, Final

QUEUES: Final[Tuple[str, ...]] = ("RANKED_SOLO_5x5", "RANKED_FLEX_SR")
ELITE_TIERS: Final[Tuple[str, ...]] = ("CHALLENGER", "GRANDMASTER", "MASTER")
TIERS: Final[Tuple[str, ...]] = (
    "DIAMOND",
    "EMERALD",
    "PLATINUM",
    "GOLD",
    "SILVER",
    "BRONZE",
    "IRON",
)
DIVISIONS: Final[Tuple[str, ...]] = ("I", "II", "III", "IV")

QUEUE_TYPE_TO_QUEUE_CODE: Final[Mapping[str, int]] = {
    "RANKED_SOLO_5x5": 420,
    "RANKED_FLEX_SR": 440,
}


def cumulative_mapping(sequence: Tuple[str, ...]) -> Mapping[str, Tuple[str, ...]]:
    return {value: sequence[: i + 1] for i, value in enumerate(sequence)}


@lru_cache(maxsize=None)
def cumulative_tier_mapping() -> Mapping[str, Tuple[str, ...]]:
    return cumulative_mapping(TIERS)


@lru_cache(maxsize=None)
def cumulative_elite_tier_mapping() -> Mapping[str, Tuple[str, ...]]:
    return cumulative_mapping(ELITE_TIERS)


@lru_cache(maxsize=None)
def cumulative_division_mapping() -> Mapping[str, Tuple[str, ...]]:
    return cumulative_mapping(DIVISIONS)