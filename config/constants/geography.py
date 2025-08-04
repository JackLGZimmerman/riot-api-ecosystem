from enum import StrEnum
from typing import Final, Mapping
from collections import defaultdict

class Continent(StrEnum):
    AMERICAS = "americas"
    EUROPE   = "europe"
    ASIA     = "asia"
    SEA      = "sea"

class Region(StrEnum):
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

REGION_TO_CONTINENT: Final[Mapping[Region, Continent]] = {
    Region.BR1:  Continent.AMERICAS,
    Region.LA1:  Continent.AMERICAS,
    Region.LA2:  Continent.AMERICAS,
    Region.NA1:  Continent.AMERICAS,
    Region.EUW1: Continent.EUROPE,
    Region.EUN1: Continent.EUROPE,
    Region.RU:   Continent.EUROPE,
    Region.TR1:  Continent.EUROPE,
    Region.ME1:  Continent.EUROPE,
    Region.JP1:  Continent.ASIA,
    Region.KR:   Continent.ASIA,
    Region.TW2:  Continent.SEA,
    Region.OC1:  Continent.SEA,
    Region.VN2:  Continent.SEA,
    Region.SG2:  Continent.SEA,
}

_temp: dict[Continent, list[Region]] = defaultdict(list)
for region, continent in REGION_TO_CONTINENT.items():
    _temp[continent].append(region)

CONTINENT_TO_REGIONS: Final[Mapping[Continent, list[Region]]] = dict(_temp)
