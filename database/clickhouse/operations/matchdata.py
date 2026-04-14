from collections.abc import Iterable

from database.clickhouse.client import get_client


def delete_by_matchids(
    table: str,
    match_ids: Iterable[str],
) -> None:
    ids = list(dict.fromkeys(str(match_id) for match_id in match_ids if match_id))
    if not ids:
        return

    get_client().command(
        f"""
        ALTER TABLE {table}
        DELETE
        WHERE has(%(match_ids)s, matchid)
        SETTINGS mutations_sync = 2
        """,
        parameters={"match_ids": ids},
    )
