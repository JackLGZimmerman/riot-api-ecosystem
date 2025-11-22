from __future__ import annotations
from typing import Final, Literal, NewType, TypedDict

# Narrow the template type so you don't accidentally stick random strings in.
URLTemplate = NewType("URLTemplate", str)

# Host fragments
__HOST_REGION = "https://{region}.api.riotgames.com"
__HOST_CONTINENT = "https://{continent}.api.riotgames.com"

# Versioned roots
__V4_LEAGUE = "/lol/league/v4"
__V4_SUMMONER = "/lol/summoner/v4"
__V5_MATCH = "/lol/match/v5"

# Literal union of top-level groups (nice for function params, etc.)
RiotEndpoints = Literal["league", "summoner", "match"]

# ---- Typed schemas for each group ----


class LeagueGroup(TypedDict):
    elite: URLTemplate
    by_queue_tier_division: URLTemplate


class SummonerGroup(TypedDict):
    by_puuid: URLTemplate


class MatchGroup(TypedDict):
    by_puuid: URLTemplate
    by_match_id: URLTemplate
    timeline_by_match_id: URLTemplate


class Endpoints(TypedDict):
    league: LeagueGroup
    summoner: SummonerGroup
    match: MatchGroup


# ---- Fully typed constant ----
ENDPOINTS: Final[Endpoints] = {
    "league": {
        "elite": URLTemplate(
            f"{__HOST_REGION}{__V4_LEAGUE}/{{elite_tier}}leagues/by-queue/{{queue}}"
            f"?api_key={{api_key}}"
        ),
        "by_queue_tier_division": URLTemplate(
            f"{__HOST_REGION}{__V4_LEAGUE}/entries/{{queue}}/{{tier}}/{{division}}"
            f"?page={{page}}&api_key={{api_key}}"
        ),
    },
    "summoner": {
        "by_puuid": URLTemplate(
            f"{__HOST_REGION}{__V4_SUMMONER}/summoners/by-puuid/{{puuid}}?api_key={{api_key}}"
        ),
    },
    "match": {
        "by_puuid": URLTemplate(
            f"{__HOST_CONTINENT}{__V5_MATCH}/matches/by-puuid/{{puuid}}/ids"
            f"?startTime={{startTime}}&endTime={{endTime}}"
            f"&type={{type}}&queue={{queue}}&start={{start}}&count=100&api_key={{api_key}}"
        ),
        "by_match_id": URLTemplate(
            f"{__HOST_CONTINENT}{__V5_MATCH}/matches/{{matchId}}?api_key={{api_key}}"
        ),
        "timeline_by_match_id": URLTemplate(
            f"{__HOST_CONTINENT}{__V5_MATCH}/matches/{{matchId}}/timeline?api_key={{api_key}}"
        ),
    },
}
