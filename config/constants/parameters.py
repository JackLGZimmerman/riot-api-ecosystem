from typing import TypeVar, Mapping
from enum import StrEnum

# Define a generic enum type variable
E = TypeVar("E", bound=StrEnum)


class Queues(StrEnum):
    RANKED_SOLO_5x5 = "RANKED_SOLO_5x5"
    RANKED_FLEX_SR = "RANKED_FLEX_SR"


QUEUE_TYPE_TO_QUEUE_CODE: Mapping[Queues, int] = {
    Queues.RANKED_SOLO_5x5: 420,
    Queues.RANKED_FLEX_SR: 440,
}


class EliteTiers(StrEnum):
    CHALLENGER = "CHALLENGER"
    GRANDMASTER = "GRANDMASTER"
    MASTER = "MASTER"


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
