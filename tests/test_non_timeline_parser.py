from __future__ import annotations

from types import SimpleNamespace

from app.services.riot_api_client.parsers.non_timeline import (
    ObjectivesParser,
    ParticipantStatsParser,
)


def test_objectives_parser_includes_atakhan() -> None:
    info = SimpleNamespace(
        teams=[
            SimpleNamespace(
                teamId=100,
                objectives=SimpleNamespace(
                    atakhan=SimpleNamespace(first=True, kills=1),
                ),
            )
        ]
    )

    assert ObjectivesParser().parse(info, "EUW1_1") == [
        {
            "matchId": "EUW1_1",
            "teamId": 100,
            "objectiveType": "atakhan",
            "first": True,
            "kills": 1,
        }
    ]


class _Participant:
    PlayerBehavior = None

    def model_dump(self, exclude: set[str]) -> dict[str, object]:
        return {
            "teamId": 100,
            "puuid": "PUUID",
            "participantId": 1,
            "visionScore": 300,
            "wardsPlaced": 301,
            "wardsKilled": 302,
            "allInPings": 303,
            "retreatPings": 304,
            "unrealKills": 300,
        }


def test_participant_stats_parser_preserves_uint16_values_above_255() -> None:
    rows = ParticipantStatsParser().parse([_Participant()], "NA1_1")

    assert len(rows) == 1
    row = rows[0]
    assert row["visionScore"] == 300
    assert row["wardsPlaced"] == 301
    assert row["wardsKilled"] == 302
    assert row["allInPings"] == 303
    assert row["retreatPings"] == 304
    assert row["unrealKills"] == 255
