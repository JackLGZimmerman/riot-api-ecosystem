import time
import asyncio
import logging
from typing import NamedTuple
from collections.abc import AsyncIterator
from uuid import UUID
from app.models.riot.league import MinifiedLeagueEntryDTO
from database.clickhouse.client import get_client
from database.clickhouse.operations.utils import (
    delete_timestamp_for_run,
    delete_timestamps_except_run,
    record_timestamp,
)

logger = logging.getLogger(__name__)

PLAYERS_TABLE = "game_data.players"
PLAYERS_SNAPSHOT_TIMESTAMP_NAME = "players_snapshot_ts"
PLAYERS_INSERT_BATCH_SIZE = 10_000

PLAYERS_COLS = [
    "run_id",
    "puuid",
    "queue_type",
    "tier",
    "division",
    "wins",
    "losses",
    "region",
    "updated_at",
]


class PlayerKeyRow(NamedTuple):
    puuid: str
    queue_type: str
    region: str


def _insert_rows(rows: list[tuple]) -> None:
    if not rows:
        return
    get_client().insert(
        table=PLAYERS_TABLE,
        data=rows,
        column_names=PLAYERS_COLS,
    )
    logger.debug("Inserted %d rows into %s", len(rows), PLAYERS_TABLE)


async def insert_players_stream_in_batches(
    players: AsyncIterator[MinifiedLeagueEntryDTO],
    ts: int,
    *,
    run_id: UUID,
    batch_size: int = PLAYERS_INSERT_BATCH_SIZE,
    flush_interval_s: float = 5.0,
) -> None:
    loop = asyncio.get_running_loop()
    batch: list[tuple] = []
    last_flush = time.monotonic()

    async for p in players:
        batch.append(
            (
                run_id,
                p.puuid,
                p.queueType,
                p.tier,
                p.division,
                int(p.wins),
                int(p.losses),
                p.region,
                ts,
            )
        )

        if (
            len(batch) >= batch_size
            or (time.monotonic() - last_flush) >= flush_interval_s
        ):
            rows = batch
            batch = []
            await loop.run_in_executor(None, _insert_rows, rows)
            last_flush = time.monotonic()

    if batch:
        await loop.run_in_executor(None, _insert_rows, batch)


def delete_partial_players_run(run_id: UUID) -> None:
    query = """
        ALTER TABLE game_data.players
        DELETE WHERE run_id = %(run_id)s
    """

    get_client().command(query, parameters={"run_id": run_id})


def upsert_players_snapshot_ts(ts: int, run_id: UUID) -> None:
    record_timestamp(PLAYERS_SNAPSHOT_TIMESTAMP_NAME, run_id, ts)


def delete_failed_players_snapshot_ts(run_id: UUID) -> None:
    delete_timestamp_for_run(PLAYERS_SNAPSHOT_TIMESTAMP_NAME, run_id)


def delete_old_players_snapshot_ts(run_id: UUID) -> None:
    delete_timestamps_except_run(PLAYERS_SNAPSHOT_TIMESTAMP_NAME, run_id)


def load_players() -> list[PlayerKeyRow]:
    query = """
    SELECT
        DISTINCT
        puuid,
        queue_type,
        region
    FROM game_data.players
    """

    result = get_client().query(query)

    return [
        PlayerKeyRow(
            puuid=row[0].decode("utf-8").rstrip("\x00")
            if isinstance(row[0], (bytes, bytearray))
            else row[0],
            queue_type=row[1],
            region=row[2],
        )
        for row in result.result_rows
    ]
