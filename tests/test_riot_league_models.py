from __future__ import annotations

from app.core.config.constants import Queues, Region
from app.models.riot.league import LeagueListDTO, MinifiedLeagueEntryDTO


def test_elite_league_list_accepts_payload_without_metadata() -> None:
    payload = {
        "tier": "CHALLENGER",
        "queue": "RANKED_SOLO_5x5",
        "entries": [
            {
                "puuid": "player-puuid",
                "leaguePoints": 1000,
                "rank": "I",
                "wins": 10,
                "losses": 5,
                "veteran": False,
                "inactive": False,
                "freshBlood": True,
                "hotStreak": False,
            },
        ],
    }

    dto = LeagueListDTO.model_validate(payload)
    entries = MinifiedLeagueEntryDTO.from_list(dto, region=Region.EUW1)

    assert entries[0].puuid == "player-puuid"
    assert entries[0].queueType == Queues.RANKED_SOLO_5x5
    assert entries[0].tier == "CHALLENGER"
    assert entries[0].division == "I"
    assert entries[0].region == Region.EUW1
