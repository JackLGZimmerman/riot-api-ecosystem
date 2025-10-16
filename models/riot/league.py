from typing import List, Optional, Union
from models.base import BaseORJSONModel
from pydantic import ConfigDict, BaseModel, computed_field, TypeAdapter

class MiniSeriesDTO(BaseORJSONModel):
    model_config = ConfigDict(slots=True)
    losses: int
    progress: str
    target: int
    wins: int

class LeagueItemDTO(BaseORJSONModel):
    model_config = ConfigDict(slots=True)
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

class LeagueListDTO(BaseORJSONModel):
    model_config = ConfigDict(slots=True)
    leagueId: str
    entries: List[LeagueItemDTO]
    tier: str
    name: str
    queue: str

class LeagueEntryDTO(BaseORJSONModel):
    model_config = ConfigDict(slots=True)
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

class MinifiedLeagueEntryDTO(BaseModel):
    model_config = ConfigDict(slots=True)
    puuid: str
    queueType: str
    tier: str
    rank: str
    wins: int
    losses: int
    region: str

    @classmethod
    def from_entry(
        cls,
        entry: "LeagueEntryDTO",
        *,
        region: str,
    ) -> "MinifiedLeagueEntryDTO":
        data = {
            "puuid": entry.puuid,
            "queueType": entry.queueType,
            "tier": entry.tier,
            "rank": entry.rank,
            "wins": entry.wins,
            "losses": entry.losses,
            "region": region,
        }
        return cls.model_validate(data)

    @classmethod
    def from_list(
        cls,
        dto: "LeagueListDTO",
        *,
        region: str,
    ) -> List["MinifiedLeagueEntryDTO"]:
        payload = [
            {
                "puuid": e.puuid,
                "queueType": dto.queue,
                "tier": dto.tier,
                "rank": e.rank,
                "wins": e.wins,
                "losses": e.losses,
                "region": region,
            }
            for e in dto.entries
        ]
        adapter = TypeAdapter(List[MinifiedLeagueEntryDTO])
        return adapter.validate_python(payload)