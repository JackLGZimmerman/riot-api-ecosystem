from typing import Iterable

from database.clickhouse.client import get_client

MATCHID_FULL_TABLES: tuple[str, ...] = (
    "game_data.metadata",
    "game_data.info",
    "game_data.bans",
    "game_data.feats",
    "game_data.objectives",
    "game_data.participant_stats",
    "game_data.participant_challenges",
    "game_data.participant_perk_values",
    "game_data.participant_perk_ids",
    "game_data.tl_participant_stats",
    "game_data.tl_building_kill",
    "game_data.tl_champion_kill",
    "game_data.tl_champion_special_kill",
    "game_data.tl_dragon_soul_given",
    "game_data.tl_elite_monster_kill",
    "game_data.tl_payload_event",
    "game_data.tl_turret_plate_destroyed",
    "game_data.tl_ck_victim_damage_dealt",
    "game_data.tl_ck_victim_damage_received",
)


def ensure_matchid_full_schema() -> None:
    client = get_client()
    for table in MATCHID_FULL_TABLES:
        client.command(
            f"""
            ALTER TABLE {table}
            ADD COLUMN IF NOT EXISTS matchidfull String
            AFTER matchid
            """
        )


def delete_by_matchids(
    table: str,
    match_ids: Iterable[str],
) -> None:
    ids = list(dict.fromkeys(str(match_id) for match_id in match_ids if match_id))
    if not ids:
        return

    command = f"""
        ALTER TABLE {table}
        DELETE
        WHERE has(%(match_ids)s, matchidfull)
    """
    get_client().command(
        command,
        parameters={
            "match_ids": ids,
        },
    )
