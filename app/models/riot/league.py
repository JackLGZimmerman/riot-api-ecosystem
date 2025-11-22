from typing import List, Optional, TypedDict
from pydantic import BaseModel, TypeAdapter


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
    rank: str
    wins: int
    losses: int
    region: str


class MinifiedLeagueEntryDTO(BaseModel):
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
        data: EntryPayload = {
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
        payload: List[EntryPayload] = [
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
