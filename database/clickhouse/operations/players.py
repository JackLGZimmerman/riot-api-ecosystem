import asyncio
import logging
from typing import AsyncIterator

from app.models.riot.league import MinifiedLeagueEntryDTO
from app.worker.pipelines.matchids_orchestrator import PlayerKeyRow
from database.clickhouse.client import get_client

logger = logging.getLogger(__name__)


def _insert_rows(rows: list[dict[str, object]]) -> None:
    if not rows:
        return

    columns = list(rows[0].keys())
    data = [tuple(row[col] for col in columns) for row in rows]

    get_client().insert(
        table="players",
        data=data,
        column_names=columns,
    )

    logger.info(f"Inserted {len(rows)} rows into ClickHouse")


async def insert_players_stream_in_batches(
    players: AsyncIterator[MinifiedLeagueEntryDTO],
    ts,
    *,
    batch_size: int = 20000,
    flush_interval_s: float = 1.0,
) -> None:
    batch: list[dict] = []
    last_flush = asyncio.get_running_loop().time()

    async for p in players:
        batch.append(
            {
                "puuid": p.puuid,
                "queue_type": p.queueType,
                "tier": p.tier,
                "division": p.division,
                "wins": int(p.wins),
                "losses": int(p.losses),
                "region": p.region,
                "updated_at": ts,
            }
        )

        now = asyncio.get_running_loop().time()
        if len(batch) >= batch_size or (now - last_flush) >= flush_interval_s:
            await asyncio.to_thread(_insert_rows, batch)
            batch.clear()
            last_flush = now

    if batch:
        await asyncio.to_thread(_insert_rows, batch)


def load_players() -> list[PlayerKeyRow]:
    query = """
    SELECT
        puuid,
        queue_type,
        region
    FROM game_data.players
    LIMIT 1 BY puuid, queue_type
    """

    result = get_client().query(query)

    return [
        PlayerKeyRow(puuid=row[0], queue_type=row[1], region=row[2])
        for row in result.result_rows
    ]
