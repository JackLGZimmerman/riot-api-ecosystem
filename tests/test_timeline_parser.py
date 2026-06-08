"""Equivalence/golden tests for the timeline event parsers.

These pin the exact per-row output of every timeline parser so the
EventTypeParser de-duplication refactor (folding the per-event `parse`
overrides into the base via DEFAULTS / EMPTY_LIST_FIELDS / _flatten_position)
cannot silently change parsed shape — the raw ClickHouse columns depend on it.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from app.services.riot_api_client.parsers.models.timeline import Position
from app.services.riot_api_client.parsers.timeline import (
    BuildingKillParser,
    ChampionKillParser,
    ChampionSpecialKillParser,
    DragonSoulGivenParser,
    EliteMonsterKillParser,
    GameEndParser,
    LevelUpParser,
    TurretPlateDestroyedParser,
    VictimDamageDealtParser,
)

MATCH_ID = "EUW1_1"
# frame.timestamp 60000 -> nearest_frame_timestamp == 60000
FRAME_TS = 60000


def _frame(*events: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(timestamp=FRAME_TS, events=list(events))


def _parse(parser: Any, *events: dict[str, Any]) -> list[dict[str, Any]]:
    return parser.parse([_frame(*events)], MATCH_ID)


def test_champion_special_kill_multikill_present_and_absent() -> None:
    rows = _parse(
        ChampionSpecialKillParser(),
        {
            "type": "CHAMPION_SPECIAL_KILL",
            "timestamp": 1000,
            "killType": "KILL_FIRST_BLOOD",
            "killerId": 2,
            "position": {"x": 10, "y": 20},
            "multiKillLength": 3,
        },
        {
            "type": "CHAMPION_SPECIAL_KILL",
            "timestamp": 1001,
            "killType": "KILL_ACE",
            "killerId": 4,
            "position": {"x": 1, "y": 2},
        },
    )
    assert rows == [
        {
            "type": "CHAMPION_SPECIAL_KILL",
            "timestamp": 1000,
            "killType": "KILL_FIRST_BLOOD",
            "killerId": 2,
            "multiKillLength": 3,
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
            "position_x": 10,
            "position_y": 20,
        },
        {
            "type": "CHAMPION_SPECIAL_KILL",
            "timestamp": 1001,
            "killType": "KILL_ACE",
            "killerId": 4,
            "multiKillLength": None,
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
            "position_x": 1,
            "position_y": 2,
        },
    ]


def test_elite_monster_kill_assisting_and_subtype_variants() -> None:
    rows = _parse(
        EliteMonsterKillParser(),
        {
            "type": "ELITE_MONSTER_KILL",
            "timestamp": 2000,
            "killerId": 1,
            "killerTeamId": 100,
            "monsterType": "DRAGON",
            "monsterSubType": "FIRE_DRAGON",
            "bounty": 50,
            "position": {"x": 3, "y": 4},
            "assistingParticipantIds": [2, 3],
        },
        {
            "type": "ELITE_MONSTER_KILL",
            "timestamp": 2001,
            "killerId": 1,
            "killerTeamId": 200,
            "monsterType": "BARON_NASHOR",
            "bounty": 0,
            "position": {"x": 5, "y": 6},
        },
        {
            "type": "ELITE_MONSTER_KILL",
            "timestamp": 2002,
            "killerId": 1,
            "killerTeamId": 200,
            "monsterType": "RIFTHERALD",
            "bounty": 0,
            "position": {"x": 7, "y": 8},
            "assistingParticipantIds": None,
        },
    )
    assert rows == [
        {
            "type": "ELITE_MONSTER_KILL",
            "timestamp": 2000,
            "killerId": 1,
            "killerTeamId": 100,
            "monsterType": "DRAGON",
            "monsterSubType": "FIRE_DRAGON",
            "bounty": 50,
            "assistingParticipantIds": [2, 3],
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
            "position_x": 3,
            "position_y": 4,
        },
        {
            "type": "ELITE_MONSTER_KILL",
            "timestamp": 2001,
            "killerId": 1,
            "killerTeamId": 200,
            "monsterType": "BARON_NASHOR",
            "monsterSubType": None,
            "bounty": 0,
            "assistingParticipantIds": [],
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
            "position_x": 5,
            "position_y": 6,
        },
        {
            "type": "ELITE_MONSTER_KILL",
            "timestamp": 2002,
            "killerId": 1,
            "killerTeamId": 200,
            "monsterType": "RIFTHERALD",
            "monsterSubType": None,
            "bounty": 0,
            "assistingParticipantIds": [],
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
            "position_x": 7,
            "position_y": 8,
        },
    ]


def test_building_kill_tower_and_assisting_variants() -> None:
    rows = _parse(
        BuildingKillParser(),
        {
            "type": "BUILDING_KILL",
            "timestamp": 3000,
            "bounty": 100,
            "buildingType": "TOWER_BUILDING",
            "killerId": 5,
            "laneType": "MID_LANE",
            "teamId": 200,
            "towerType": "OUTER_TURRET",
            "assistingParticipantIds": [6],
            "position": {"x": 7, "y": 8},
        },
        {
            "type": "BUILDING_KILL",
            "timestamp": 3001,
            "bounty": 0,
            "buildingType": "INHIBITOR_BUILDING",
            "killerId": 7,
            "laneType": "TOP_LANE",
            "teamId": 100,
            "position": {"x": 9, "y": 10},
        },
    )
    assert rows == [
        {
            "type": "BUILDING_KILL",
            "timestamp": 3000,
            "bounty": 100,
            "buildingType": "TOWER_BUILDING",
            "killerId": 5,
            "laneType": "MID_LANE",
            "teamId": 200,
            "towerType": "OUTER_TURRET",
            "assistingParticipantIds": [6],
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
            "position_x": 7,
            "position_y": 8,
        },
        {
            "type": "BUILDING_KILL",
            "timestamp": 3001,
            "bounty": 0,
            "buildingType": "INHIBITOR_BUILDING",
            "killerId": 7,
            "laneType": "TOP_LANE",
            "teamId": 100,
            "towerType": None,
            "assistingParticipantIds": [],
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
            "position_x": 9,
            "position_y": 10,
        },
    ]


def test_game_end_drops_type_and_defaults_game_id() -> None:
    rows = _parse(
        GameEndParser(),
        {
            "type": "GAME_END",
            "timestamp": 4000,
            "winningTeam": 100,
            "realTimestamp": 1700000000000,
            "gameId": 123,
        },
        {
            "type": "GAME_END",
            "timestamp": 4001,
            "winningTeam": 200,
            "realTimestamp": 1700000000001,
        },
    )
    assert rows == [
        {
            "timestamp": 4000,
            "winningTeam": 100,
            "realTimestamp": 1700000000000,
            "gameId": 123,
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
        },
        {
            "timestamp": 4001,
            "winningTeam": 200,
            "realTimestamp": 1700000000001,
            "gameId": None,
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
        },
    ]


def test_dragon_soul_given_keeps_type_no_position() -> None:
    rows = _parse(
        DragonSoulGivenParser(),
        {
            "type": "DRAGON_SOUL_GIVEN",
            "timestamp": 5000,
            "name": "Infernal",
            "teamId": 100,
        },
    )
    assert rows == [
        {
            "type": "DRAGON_SOUL_GIVEN",
            "timestamp": 5000,
            "name": "Infernal",
            "teamId": 100,
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
        }
    ]


def test_level_up_drops_type_no_position() -> None:
    rows = _parse(
        LevelUpParser(),
        {"type": "LEVEL_UP", "timestamp": 6000, "level": 2, "participantId": 3},
    )
    assert rows == [
        {
            "timestamp": 6000,
            "level": 2,
            "participantId": 3,
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
        }
    ]


def test_turret_plate_destroyed_position_object_branch() -> None:
    rows = _parse(
        TurretPlateDestroyedParser(),
        {
            "type": "TURRET_PLATE_DESTROYED",
            "timestamp": 7000,
            "killerId": 1,
            "laneType": "MID_LANE",
            "teamId": 200,
            "position": Position(x=11, y=12),
        },
    )
    assert rows == [
        {
            "type": "TURRET_PLATE_DESTROYED",
            "timestamp": 7000,
            "killerId": 1,
            "laneType": "MID_LANE",
            "teamId": 200,
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
            "position_x": 11,
            "position_y": 12,
        }
    ]


def test_champion_kill_strips_damage_builds_event_id() -> None:
    rows = _parse(
        ChampionKillParser(),
        {
            "type": "CHAMPION_KILL",
            "timestamp": 8000,
            "bounty": 300,
            "killStreakLength": 1,
            "killerId": 2,
            "victimId": 5,
            "shutdownBounty": 0,
            "position": {"x": 13, "y": 14},
            "assistingParticipantIds": [3, 4],
            "victimDamageDealt": [{"basic": True}],
            "victimDamageReceived": [{"basic": False}],
            "victimTeamfightDamageDealt": [{"basic": True}],
            "victimTeamfightDamageReceived": [{"basic": False}],
        },
        {
            "type": "CHAMPION_KILL",
            "timestamp": 8001,
            "bounty": 0,
            "killStreakLength": 0,
            "killerId": 6,
            "victimId": 7,
            "shutdownBounty": 0,
            "position": {"x": 1, "y": 1},
        },
    )
    assert rows == [
        {
            "type": "CHAMPION_KILL",
            "timestamp": 8000,
            "bounty": 300,
            "killStreakLength": 1,
            "killerId": 2,
            "victimId": 5,
            "shutdownBounty": 0,
            "assistingParticipantIds": [3, 4],
            "champion_kill_event_id": "EUW1_1:8000:2:5",
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
            "position_x": 13,
            "position_y": 14,
        },
        {
            "type": "CHAMPION_KILL",
            "timestamp": 8001,
            "bounty": 0,
            "killStreakLength": 0,
            "killerId": 6,
            "victimId": 7,
            "shutdownBounty": 0,
            "assistingParticipantIds": [],
            "champion_kill_event_id": "EUW1_1:8001:6:7",
            "frame_timestamp": FRAME_TS,
            "matchId": MATCH_ID,
            "position_x": 1,
            "position_y": 1,
        },
    ]


def test_victim_damage_dealt_emits_one_row_per_instance() -> None:
    rows = _parse(
        VictimDamageDealtParser(),
        {
            "type": "CHAMPION_KILL",
            "timestamp": 9000,
            "killerId": 2,
            "victimId": 5,
            "victimDamageDealt": [
                {
                    "basic": True,
                    "magicDamage": 0,
                    "name": "Ahri",
                    "participantId": 2,
                    "physicalDamage": 100,
                    "spellName": "AhriQ",
                    "spellSlot": 1,
                    "trueDamage": 0,
                    "type": "OTHER",
                },
                {
                    "basic": False,
                    "magicDamage": 50,
                    "name": "Ahri",
                    "participantId": 2,
                    "physicalDamage": 0,
                    "spellName": "AhriW",
                    "spellSlot": 2,
                    "trueDamage": 0,
                    "type": "OTHER",
                },
            ],
        },
    )
    cid = "EUW1_1:9000:2:5"
    assert rows == [
        {
            "basic": True,
            "magicDamage": 0,
            "name": "Ahri",
            "participantId": 2,
            "physicalDamage": 100,
            "spellName": "AhriQ",
            "spellSlot": 1,
            "trueDamage": 0,
            "type": "OTHER",
            "matchId": MATCH_ID,
            "frame_timestamp": FRAME_TS,
            "timestamp": 9000,
            "direction": "DEALT",
            "champion_kill_event_id": cid,
            "idx": 0,
        },
        {
            "basic": False,
            "magicDamage": 50,
            "name": "Ahri",
            "participantId": 2,
            "physicalDamage": 0,
            "spellName": "AhriW",
            "spellSlot": 2,
            "trueDamage": 0,
            "type": "OTHER",
            "matchId": MATCH_ID,
            "frame_timestamp": FRAME_TS,
            "timestamp": 9000,
            "direction": "DEALT",
            "champion_kill_event_id": cid,
            "idx": 1,
        },
    ]
