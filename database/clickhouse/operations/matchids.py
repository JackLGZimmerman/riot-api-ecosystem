import asyncio
from typing import AsyncIterator, Iterable

from database.clickhouse.client import get_client


def load_matchid_puuid_timestamp() -> int:
    query = """
        SELECT stored_at FROM game_data.data_timestamps
        WHERE name = 'matchids_puuids_ts'
    """

    result = get_client().query(query)
    return int(result.result_rows[0][0])


def load_matchids() -> list[str]:
    query = """
        SELECT matchid FROM game_data.matchids;
    """

    rows = get_client().query(query).result_rows
    return [row[0] for row in rows]


def load_matchid_puuids() -> list[str]:
    query = """
        SELECT puuids FROM game_data.matchids;
    """

    rows = get_client().query(query).result_rows
    return [row[0] for row in rows]


def insert_puuid_timestamp(ts: int) -> None:
    get_client().insert(
        table="data_timestamps",
        data=[["matchids_puuids_ts", ts]],
        column_names=["name", "stored_at"],
    )


def insert_puuids_in_batches(
    puuids: Iterable[str], *, batch_size: int = 50_000
) -> None:
    client = get_client()
    client.query("TRUNCATE TABLE game_data.matchid_puuids")

    batch: list[list[str]] = []
    for p in puuids:
        batch.append([p])
        if len(batch) >= batch_size:
            client.insert("game_data.matchid_puuids", batch, column_names=["puuid"])
            batch.clear()

    if batch:
        client.insert("game_data.matchid_puuids", batch, column_names=["puuid"])


async def insert_matchids_stream_in_batches(
    match_ids_stream: AsyncIterator[list[str]],
    *,
    buffer_size: int = 200_000,
) -> None:
    buf: list[str] = []

    async for batch in match_ids_stream:
        if not batch:
            continue

        buf.extend(batch)

        if len(buf) >= buffer_size:
            data = [(mid,) for mid in buf]
            await asyncio.to_thread(
                get_client().insert,
                "game_data.matchids",
                data,
                ["matchid"],
            )
            buf.clear()

    if buf:
        data = [(mid,) for mid in buf]
        await asyncio.to_thread(
            get_client().insert,
            "game_data.matchids",
            data,
            ["matchid"],
        )
