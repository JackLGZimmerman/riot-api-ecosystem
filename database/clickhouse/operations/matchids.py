import asyncio
from typing import AsyncIterator, Iterable
import time
from database.clickhouse.client import get_client
from app.core.config import settings
from uuid import UUID

PUUID_DATA_TIMESTAMP_NAME = "matchids_puuids_ts"
_CH_EXECUTOR = settings.threadpool_executor_clickhouse


def load_matchid_puuid_ts() -> int:
    query = """
        SELECT stored_at
        FROM game_data.data_timestamps
        WHERE name = %(name)s
    """
    res = get_client().query(query, parameters={"name": PUUID_DATA_TIMESTAMP_NAME})
    return int(res.result_rows[0][0]) if res.result_rows else 0


def load_matchids() -> list[str]:
    query = """
        SELECT matchid FROM game_data.matchids;
    """

    rows = get_client().query(query).result_rows
    return [row[0] for row in rows]


def load_matchid_puuids() -> list[str]:
    query = "SELECT puuid FROM game_data.matchid_puuids"
    rows = get_client().query(query).result_rows
    return [row[0] for row in rows]


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


def delete_matchid_puuids(run_id: UUID) -> None:
    get_client().command(
        """
        ALTER TABLE game_data.matchid_puuids
        DELETE WHERE run_id = %(run_id)s
    """,
        parameters={"run_id": run_id},
    )


def delete_matchids(run_id: UUID) -> None:
    get_client().command(
        """
        ALTER TABLE game_data.matchids
        DELETE WHERE run_id = %(run_id)s
    """,
        parameters={"run_id": run_id},
    )


def upsert_puuid_timestamp(ts: int, run_id: UUID) -> None:
    client = get_client()
    client.insert(
        table="game_data.data_timestamps",
        data=[(run_id, PUUID_DATA_TIMESTAMP_NAME, ts)],
        column_names=["run_id", "name", "stored_at"],
    )


def insert_puuids_in_batches(
    puuids: Iterable[str],
    run_id: UUID,
    *,
    batch_size: int = 50_000,
) -> None:
    client = get_client()

    batch: list[tuple] = []
    for p in puuids:
        batch.append((run_id, p))

        if len(batch) >= batch_size:
            client.insert(
                table="game_data.matchid_puuids",
                data=batch,
                column_names=["run_id", "puuid"],
            )
            batch.clear()

    if batch:
        client.insert(
            table="game_data.matchid_puuids",
            data=batch,
            column_names=["run_id", "puuid"],
        )


async def insert_matchids_stream_in_batches(
    match_ids_stream: AsyncIterator[list[str]],
    run_id: UUID,
    *,
    buffer_size: int = 200_000,
    flush_interval_s: float = 1.0,
) -> None:
    loop = asyncio.get_running_loop()
    buf: list[str] = []
    last_flush = time.monotonic()

    async for batch in match_ids_stream:
        if not batch:
            continue

        buf.extend(batch)

        if (
            len(buf) >= buffer_size
            or (time.monotonic() - last_flush) >= flush_interval_s
        ):
            rows = [(run_id, mid) for mid in buf]
            buf = []

            await loop.run_in_executor(
                _CH_EXECUTOR,
                get_client().insert,
                "game_data.matchids",
                rows,
                ["run_id", "matchid"],
            )
            last_flush = time.monotonic()

    if buf:
        rows = [(run_id, mid) for mid in buf]
        await loop.run_in_executor(
            _CH_EXECUTOR,
            get_client().insert,
            "game_data.matchids",
            rows,
            ["run_id", "matchid"],
        )
