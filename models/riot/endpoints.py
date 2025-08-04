from pydantic import BaseModel

class LeagueEndpoints(BaseModel):
    by_summoner_id:         str
    challenger:             str
    grandmaster:            str
    master:                 str
    by_queue_tier_division: str

class SummonerEndpoints(BaseModel):
    by_name:        str
    by_puuid:       str
    by_summoner_id: str
    by_account_id:  str

class MatchEndpoints(BaseModel):
    by_puuid:             str
    by_match_id:          str
    timeline_by_match_id: str

class EndpointsSchema(BaseModel):
    league:   LeagueEndpoints
    summoner: SummonerEndpoints
    match:    MatchEndpoints