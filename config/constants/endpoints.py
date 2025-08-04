from typing import Final
from models.riot.endpoints import EndpointsSchema, LeagueEndpoints, SummonerEndpoints, MatchEndpoints

__HOST_REGION    = "https://{region}.api.riotgames.com"
__HOST_CONTINENT = "https://{continent}.api.riotgames.com"

__V4_LEAGUE   = "/lol/league/v4"
__V4_SUMMONER = "/lol/summoner/v4"
__V5_MATCH    = "/lol/match/v5"

ENDPOINTS: Final[EndpointsSchema] = EndpointsSchema(
    league=LeagueEndpoints(
        by_summoner_id         = f"{__HOST_REGION}{__V4_LEAGUE}/entries/by-summoner/{{summonerId}}?api_key={{api_key}}",
        challenger             = f"{__HOST_REGION}{__V4_LEAGUE}/challengerleagues/by-queue/{{queue}}?api_key={{api_key}}",
        grandmaster            = f"{__HOST_REGION}{__V4_LEAGUE}/grandmasterleagues/by-queue/{{queue}}?api_key={{api_key}}",
        master                 = f"{__HOST_REGION}{__V4_LEAGUE}/masterleagues/by-queue/{{queue}}?api_key={{api_key}}",
        by_queue_tier_division = (
            f"{__HOST_REGION}{__V4_LEAGUE}/entries/{{queue}}/{{tier}}/{{division}}"
            f"?page={{page}}&api_key={{api_key}}"
        ),
    ),
    summoner=SummonerEndpoints(
        by_name        = f"{__HOST_REGION}{__V4_SUMMONER}/summoners/by-name/{{summonerName}}?api_key={{api_key}}",
        by_puuid       = f"{__HOST_REGION}{__V4_SUMMONER}/summoners/by-puuid/{{puuid}}?api_key={{api_key}}",
        by_summoner_id = f"{__HOST_REGION}{__V4_SUMMONER}/summoners/{{summonerId}}?api_key={{api_key}}",
        by_account_id  = f"{__HOST_REGION}{__V4_SUMMONER}/summoners/by-account/{{accountId}}?api_key={{api_key}}",
    ),
    match=MatchEndpoints(
        by_puuid             = (
            f"{__HOST_CONTINENT}{__V5_MATCH}/matches/by-puuid/{{puuid}}/ids"
            f"?startTime={{startTime}}&endTime={{endTime}}"
            f"&type={{type}}&queue={{queue}}&start={{start}}&count=100&api_key={{api_key}}"
        ),
        by_match_id          = f"{__HOST_CONTINENT}{__V5_MATCH}/matches/{{matchId}}?api_key={{api_key}}",
        timeline_by_match_id = f"{__HOST_CONTINENT}{__V5_MATCH}/matches/{{matchId}}/timeline?api_key={{api_key}}",
    ),
)