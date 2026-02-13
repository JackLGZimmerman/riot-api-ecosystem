from __future__ import annotations

from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel, TypeAdapter

from app.core.config.constants import (
    Queues,
    Region,
)
from app.core.config.constants.parameters import (
    Divisions,
    EliteTiers,
    Tiers,
)


class MiniSeriesDTO(BaseModel):
    losses: int
    progress: str
    target: int
    wins: int


class LeagueItemDTO(BaseModel):
    freshBlood: bool
    wins: int
    miniSeries: Optional[MiniSeriesDTO] = None
    inactive: bool
    veteran: bool
    hotStreak: bool
    rank: str
    leaguePoints: int
    losses: int
    puuid: str


class LeagueListDTO(BaseModel):
    leagueId: str
    entries: List[LeagueItemDTO]
    tier: str
    name: str
    queue: str


class LeagueEntryDTO(BaseModel):
    leagueId: str
    puuid: str
    queueType: str
    tier: str
    rank: str
    leaguePoints: int
    wins: int
    losses: int
    hotStreak: bool
    veteran: bool
    freshBlood: bool
    inactive: bool
    miniSeries: Optional[MiniSeriesDTO] = None


class EntryPayload(TypedDict):
    puuid: str
    queueType: str
    tier: str
    division: str
    wins: int
    losses: int
    region: str


class MinifiedLeagueEntryDTO(BaseModel):
    puuid: str
    queueType: Queues
    tier: str
    division: str
    wins: int
    losses: int
    region: Region

    @classmethod
    def from_entry(
        cls,
        entry: LeagueEntryDTO,
        *,
        region: str,
    ) -> MinifiedLeagueEntryDTO:
        data: EntryPayload = {
            "puuid": entry.puuid,
            "queueType": entry.queueType,
            "tier": entry.tier,
            "division": entry.rank,
            "wins": entry.wins,
            "losses": entry.losses,
            "region": region,
        }
        return cls.model_validate(data)

    @classmethod
    def from_list(
        cls,
        dto: LeagueListDTO,
        *,
        region: str,
    ) -> List[MinifiedLeagueEntryDTO]:
        payload: List[EntryPayload] = [
            {
                "puuid": e.puuid,
                "queueType": dto.queue,
                "tier": dto.tier,
                "division": e.rank,
                "wins": e.wins,
                "losses": e.losses,
                "region": region,
            }
            for e in dto.entries
        ]
        adapter = TypeAdapter(List[MinifiedLeagueEntryDTO])
        return adapter.validate_python(payload)


class EliteBoundConfig(BaseModel):
    collect: bool = True
    upper: EliteTiers | None = None
    lower: EliteTiers | None = None


class BasicBoundConfig(BaseModel):
    collect: bool = True
    upper_tier: Tiers | None = None
    upper_division: Divisions | None = None
    lower_tier: Tiers | None = None
    lower_division: Divisions | None = None


EliteBoundsConfig = Dict[Queues, EliteBoundConfig]
BasicBoundsConfig = Dict[Queues, BasicBoundConfig]


_elite_bounds_adapter = TypeAdapter(EliteBoundsConfig)
_basic_bounds_adapter = TypeAdapter(BasicBoundsConfig)


def parse_elite_bounds(data: Any) -> EliteBoundsConfig:
    """
    Validate and coerce an incoming elite-bounds payload into
    a dict[Queues, EliteBoundConfig].
    """
    return _elite_bounds_adapter.validate_python(data)


def parse_basic_bounds(data: Any) -> BasicBoundsConfig:
    """
    Validate and coerce an incoming basic-bounds payload into
    a dict[Queues, BasicBoundConfig].
    """
    return _basic_bounds_adapter.validate_python(data)


ELITE_BOUNDS: EliteBoundsConfig = {
    Queues.RANKED_SOLO_5x5: EliteBoundConfig(
        collect=True,
        upper=EliteTiers.CHALLENGER,
        lower=EliteTiers.CHALLENGER,
    ),
    Queues.RANKED_FLEX_SR: EliteBoundConfig(
        collect=False,
        upper=EliteTiers.CHALLENGER,
        lower=EliteTiers.CHALLENGER,
    ),
}

BASIC_BOUNDS: BasicBoundsConfig = {
    Queues.RANKED_SOLO_5x5: BasicBoundConfig(
        collect=False,
        upper_tier=Tiers.DIAMOND,
        upper_division=Divisions.I,
        lower_tier=Tiers.DIAMOND,
        lower_division=Divisions.I,
    ),
    Queues.RANKED_FLEX_SR: BasicBoundConfig(
        collect=False,
        upper_tier=Tiers.DIAMOND,
        upper_division=Divisions.I,
        lower_tier=Tiers.DIAMOND,
        lower_division=Divisions.I,
    ),
}
