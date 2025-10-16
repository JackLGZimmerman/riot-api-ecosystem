from enum import StrEnum
from typing import Final, Mapping

class Continents(StrEnum):
    AMERICAS = "americas"
    EUROPE   = "europe"
    ASIA     = "asia"
    SEA      = "sea"

class Regions(StrEnum):
    BR1  = "br1"
    LA1  = "la1"
    LA2  = "la2"
    NA1  = "na1"
    EUW1 = "euw1"
    EUN1 = "eun1"
    RU   = "ru"
    TR1  = "tr1"
    ME1  = "me1"
    JP1  = "jp1"
    KR   = "kr"
    TW2  = "tw2"
    OC1  = "oc1"
    VN2  = "vn2"
    SG2  = "sg2"

REGION_TO_CONTINENT: Final[Mapping[Regions, Continents]] = {
    Regions.BR1:  Continents.AMERICAS,
    Regions.LA1:  Continents.AMERICAS,
    Regions.LA2:  Continents.AMERICAS,
    Regions.NA1:  Continents.AMERICAS,
    Regions.EUW1: Continents.EUROPE,
    Regions.EUN1: Continents.EUROPE,
    Regions.RU:   Continents.EUROPE,
    Regions.TR1:  Continents.EUROPE,
    Regions.ME1:  Continents.EUROPE,
    Regions.JP1:  Continents.ASIA,
    Regions.KR:   Continents.ASIA,
    Regions.TW2:  Continents.SEA,
    Regions.OC1:  Continents.SEA,
    Regions.VN2:  Continents.SEA,
    Regions.SG2:  Continents.SEA,
}

CONTINENT_TO_REGIONS: Final[Mapping[Continents, tuple[Regions, ...]]] = {
    continent: tuple(r for r, c in REGION_TO_CONTINENT.items() if c is continent)
    for continent in Continents
}