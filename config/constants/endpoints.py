from typing import Final

__HOST_REGION    = "https://{region}.api.riotgames.com"
__HOST_CONTINENT = "https://{continent}.api.riotgames.com"

__V4_LEAGUE   = "/lol/league/v4"
__V4_SUMMONER = "/lol/summoner/v4"
__V5_MATCH    = "/lol/match/v5"

ENDPOINTS: Final = {
    "league": {
        "elite": (
            f"{__HOST_REGION}{__V4_LEAGUE}/{{elite_tier}}leagues/by-queue/{{queue}}"
            f"?api_key={{api_key}}"
        ),
        "by_queue_tier_division": (
            f"{__HOST_REGION}{__V4_LEAGUE}/entries/{{queue}}/{{tier}}/{{division}}"
            f"?page={{page}}&api_key={{api_key}}"
        ),
    },
    "summoner": {
        "by_puuid": (
            f"{__HOST_REGION}{__V4_SUMMONER}/summoners/by-puuid/{{puuid}}?api_key={{api_key}}"
        ),
    },
    "match": {
        "by_puuid": (
            f"{__HOST_CONTINENT}{__V5_MATCH}/matches/by-puuid/{{puuid}}/ids"
            f"?startTime={{startTime}}&endTime={{endTime}}"
            f"&type={{type}}&queue={{queue}}&start={{start}}&count=100&api_key={{api_key}}"
        ),
        "by_match_id": (
            f"{__HOST_CONTINENT}{__V5_MATCH}/matches/{{matchId}}?api_key={{api_key}}"
        ),
        "timeline_by_match_id": (
            f"{__HOST_CONTINENT}{__V5_MATCH}/matches/{{matchId}}/timeline?api_key={{api_key}}"
        ),
    },
}