import time
import asyncio
import logging
from typing import AsyncIterator, NamedTuple
from uuid import UUID
from app.core.config import settings

from app.models.riot.league import MinifiedLeagueEntryDTO
from database.clickhouse.client import get_client

logger = logging.getLogger(__name__)

_CH_EXECUTOR = settings.threadpool_executor_clickhouse

PLAYERS_TABLE = "game_data.players"

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
    logger.info("Inserted %d rows into %s", len(rows), PLAYERS_TABLE)


async def insert_players_stream_in_batches(
    players: AsyncIterator[MinifiedLeagueEntryDTO],
    ts: int,
    *,
    run_id: UUID,
    batch_size: int = 20_000,
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
            await loop.run_in_executor(_CH_EXECUTOR, _insert_rows, rows)
            last_flush = time.monotonic()

    if batch:
        await loop.run_in_executor(_CH_EXECUTOR, _insert_rows, batch)


def delete_partial_players_run(run_id: UUID) -> None:
    query = """
        ALTER TABLE game_data.players
        DELETE WHERE run_id = %(run_id)s
    """

    get_client().command(query, parameters={"run_id": run_id})


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
