from __future__ import annotations

from app.core.config.constants import Queues, Region
from app.core.config.constants.parameters import EliteTiers
from app.models.riot.league import ELITE_BOUNDS, LeagueListDTO, MinifiedLeagueEntryDTO
from app.services.riot_api_client.utils import bounded_elite_tiers


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


def test_configured_elite_bounds_collect_masters_plus() -> None:
    tiers = bounded_elite_tiers(ELITE_BOUNDS[Queues.RANKED_SOLO_5x5])

    assert tiers == [
        EliteTiers.CHALLENGER,
        EliteTiers.GRANDMASTER,
        EliteTiers.MASTER,
    ]
