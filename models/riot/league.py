from typing import List, Optional
from models.base import BaseORJSONModel
from pydantic import ConfigDict
from collections import namedtuple

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

# class MinifiedLeagueEntryDTO(BaseORJSONModel):
#     model_config = ConfigDict(slots=True)
#     puuid: str
#     queueType: str
#     tier: str
#     rank: str
#     wins: int
#     losses: int
#     region: str
#     continent: str

MinifiedLeagueEntryDTO = namedtuple("MinifiedLeagueEntryDTO", 
                                    ["puuid", 
                                     "queueType", 
                                     "tier", 
                                     "rank", 
                                     "wins", 
                                     "losses", 
                                     "region", 
                                     "continent"])