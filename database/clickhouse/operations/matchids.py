import asyncio
import logging
from typing import AsyncIterator, Iterable
from uuid import UUID

from database.clickhouse.client import get_client

PUUID_DATA_TIMESTAMP_NAME = "matchids_puuids_ts"
logger = logging.getLogger("app.services.riot_api_client.rate_limiter")


# RECOVERY-SYSTEM: deterministic per-batch insert helper used by resumable saver.
def _insert_matchids_rows(rows: list[tuple[UUID, str, str]]) -> None:
    if not rows:
        return
    logger.debug(
        "Insert matchids batch rows=%d run_id=%s",
        len(rows),
        rows[0][0],
    )
    get_client().insert(
        table="game_data.matchids",
        data=rows,
        column_names=["run_id", "matchid", "queue_type"],
    )


def insert_matchids_in_batches(
    match_rows: Iterable[tuple[str, str]],
    run_id: UUID,
    *,
    batch_size: int = 20_000,
) -> None:
    batch: list[tuple[UUID, str, str]] = []
    for match_id, queue_type in match_rows:
        batch.append((run_id, match_id, queue_type))
        if len(batch) >= batch_size:
            _insert_matchids_rows(batch)
            batch.clear()

    if batch:
        _insert_matchids_rows(batch)


async def insert_matchids_stream_in_batches(
    match_rows_stream: AsyncIterator[list[tuple[str, str]]],
    run_id: UUID,
    *,
    batch_size: int = 20_000,
) -> None:
    batch: list[tuple[UUID, str, str]] = []

    async for match_rows in match_rows_stream:
        for match_id, queue_type in match_rows:
            batch.append((run_id, match_id, queue_type))
            if len(batch) >= batch_size:
                rows = batch
                batch = []
                await asyncio.to_thread(_insert_matchids_rows, rows)

    if batch:
        await asyncio.to_thread(_insert_matchids_rows, batch)


def load_matchid_puuid_ts() -> int:
    # Anchor from the newest persisted timestamp row (robust while old-row deletes settle).
    query = """
        SELECT max(stored_at)
        FROM game_data.data_timestamps
        WHERE name = %(name)s
    """
    res = get_client().query(query, parameters={"name": PUUID_DATA_TIMESTAMP_NAME})
    if not res.result_rows:
        logger.debug("Loaded matchids puuid timestamp value=0")
        return 0
    value = res.result_rows[0][0]
    out = int(value) if value is not None else 0
    logger.debug("Loaded matchids puuid timestamp value=%s", out)
    return out


def load_matchid_puuids() -> list[tuple[str, str]]:
    query = """
        SELECT puuid, queue_type
        FROM game_data.matchid_puuids
        WHERE run_id = (
            SELECT argMax(run_id, stored_at)
            FROM game_data.data_timestamps
            WHERE name = %(name)s
        )
    """
    rows = get_client().query(
        query,
        parameters={"name": PUUID_DATA_TIMESTAMP_NAME},
    ).result_rows
    out = [(_as_text(row[0]), _as_text(row[1])) for row in rows]
    logger.debug("Loaded matchid puuids rows=%d", len(out))
    return out


def delete_failed_puuid_timestamp(run_id: UUID) -> None:
    command = """
        ALTER TABLE game_data.data_timestamps
        DELETE
        WHERE name = %(name)s
          AND run_id = %(run_id)s
    """
    get_client().command(
        command,
        parameters={
            "name": PUUID_DATA_TIMESTAMP_NAME,
            "run_id": run_id,
        },
    )
    logger.debug("Deleted failed puuid timestamp run_id=%s", run_id)


def delete_old_puuid_timestamps(run_id: UUID) -> None:
    get_client().command(
        """
        ALTER TABLE game_data.data_timestamps
        DELETE
        WHERE name = %(name)s
          AND run_id != %(run_id)s
        """,
        parameters={
            "name": PUUID_DATA_TIMESTAMP_NAME,
            "run_id": run_id,
        },
    )
    logger.debug("Deleted old puuid timestamps excluding run_id=%s", run_id)


def delete_matchid_puuids(run_id: UUID) -> None:
    get_client().command(
        """
        ALTER TABLE game_data.matchid_puuids
        DELETE WHERE run_id = %(run_id)s
    """,
        parameters={"run_id": run_id},
    )
    logger.debug("Deleted matchid_puuids run_id=%s", run_id)


def delete_matchids(run_id: UUID) -> None:
    get_client().command(
        """
        ALTER TABLE game_data.matchids
        DELETE WHERE run_id = %(run_id)s
    """,
        parameters={"run_id": run_id},
    )
    logger.debug("Deleted matchids run_id=%s", run_id)


def upsert_puuid_timestamp(ts: int, run_id: UUID) -> None:
    # RECOVERY-SYSTEM: advances cycle anchor only after all batches in cycle succeed.
    client = get_client()
    client.insert(
        table="game_data.data_timestamps",
        data=[(run_id, PUUID_DATA_TIMESTAMP_NAME, ts)],
        column_names=["run_id", "name", "stored_at"],
    )
    logger.debug("Upserted puuid timestamp run_id=%s ts=%s", run_id, ts)


def insert_puuids_in_batches(
    player_keys: Iterable[tuple[str, str]],
    run_id: UUID,
    *,
    batch_size: int = 20_000,
) -> None:
    client = get_client()

    batch: list[tuple] = []
    for puuid, queue_type in player_keys:
        batch.append((run_id, puuid, queue_type))

        if len(batch) >= batch_size:
            logger.debug(
                "Insert matchid_puuids batch rows=%d run_id=%s",
                len(batch),
                run_id,
            )
            client.insert(
                table="game_data.matchid_puuids",
                data=batch,
                column_names=["run_id", "puuid", "queue_type"],
            )
            batch.clear()

    if batch:
        logger.debug(
            "Insert matchid_puuids batch rows=%d run_id=%s",
            len(batch),
            run_id,
        )
        client.insert(
            table="game_data.matchid_puuids",
            data=batch,
            column_names=["run_id", "puuid", "queue_type"],
        )


def _as_text(value: object) -> str:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8").rstrip("\x00")
    return str(value)
