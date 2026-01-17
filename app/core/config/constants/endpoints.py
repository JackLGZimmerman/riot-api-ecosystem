from __future__ import annotations

from typing import Final, Literal, NewType, TypedDict

# Narrow the template type so you don't accidentally stick random strings in.
URLTemplate = NewType("URLTemplate", str)

__HOST_REGION = "https://{region}.api.riotgames.com"
__HOST_CONTINENT = "https://{continent}.api.riotgames.com"

__V4_LEAGUE = "/lol/league/v4"
__V4_SUMMONER = "/lol/summoner/v4"
__V5_MATCH = "/lol/match/v5"

RiotEndpoints = Literal["league", "summoner", "match"]


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


ENDPOINTS: Final[Endpoints] = {
    "league": {
        "elite": URLTemplate(
            f"{__HOST_REGION}{__V4_LEAGUE}/{{elite_tier}}leagues/by-queue/{{queue}}"
        ),
        "by_queue_tier_division": URLTemplate(
            f"{__HOST_REGION}{__V4_LEAGUE}/entries/{{queue}}/{{tier}}/{{division}}"
            f"?page={{page}}"
        ),
    },
    "summoner": {
        "by_puuid": URLTemplate(
            f"{__HOST_REGION}{__V4_SUMMONER}/summoners/by-puuid/{{puuid}}"
        ),
    },
    "match": {
        "by_puuid": URLTemplate(
            f"{__HOST_CONTINENT}{__V5_MATCH}/matches/by-puuid/{{puuid}}/ids"
            f"?startTime={{startTime}}&endTime={{endTime}}"
            f"&type={{type}}&queue={{queue}}&start={{start}}&count=100"
        ),
        "by_match_id": URLTemplate(
            f"{__HOST_CONTINENT}{__V5_MATCH}/matches/{{matchId}}"
        ),
        "timeline_by_match_id": URLTemplate(
            f"{__HOST_CONTINENT}{__V5_MATCH}/matches/{{matchId}}/timeline"
        ),
    },
}
