# file_storage.py
from __future__ import annotations

import asyncio
import aiofiles
import csv
import io
from pathlib import Path
from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, AsyncIterator, AsyncGenerator

import zstandard as zstd

# ───────────────────────────────  league_v4  ─────────────────────────────── #

async def save_league_v4(
    path: str | Path,
    rows: Iterable[Sequence[Any]]
) -> None:
    rows = list(rows)
    if not rows:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    csv_buf = io.StringIO(newline="")
    csv.writer(csv_buf).writerows(rows)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    compressor = zstd.ZstdCompressor(level=15)
    compressed = compressor.compress(csv_bytes)

    async with aiofiles.open(p, "wb") as f:
        await f.write(compressed)


async def load_league_v4(
    path: str | Path,
    *,
    indexes: Optional[List[int]] = None,
) -> AsyncIterator[List[Any]]:
    p = Path(path)

    p.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(p, "rb") as f:
        compressed = await f.read()

    decompressed = zstd.ZstdDecompressor().decompress(compressed)
    txt = io.StringIO(decompressed.decode("utf-8"))
    reader = csv.reader(txt)

    for row in reader:
        yield [row[i] for i in indexes] if indexes else row
                    

# ───────────────────────────────  match_v5_ids  ─────────────────────────────── #

async def save_match_v5_ids(
    path: str | Path,
    rows: Iterable[Sequence[Any]],
) -> None:
    rows = list(rows)
    if not rows:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    csv_buf = io.StringIO(newline="")
    csv.writer(csv_buf).writerows(rows)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    compressor = zstd.ZstdCompressor(level=15)
    compressed = compressor.compress(csv_bytes)

    async with aiofiles.open(p, "ab") as f:
        await f.write(compressed)


async def load_match_v5_ids(path: str | Path, chunk_size: int = 100) -> AsyncGenerator[list[list[Any]], None]:
    p = Path(path)

    def _stream_batches():
        dctx = zstd.ZstdDecompressor()
        with open(p, "rb") as f, dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8", newline="")
            csv_reader = csv.reader(text_stream)
            batch = []
            for row in csv_reader:
                batch.append(row)
                if len(batch) >= chunk_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    for batch in await asyncio.to_thread(lambda: list(_stream_batches())):
        yield batch

# ───────────────────────────────  match_v5_collected_players  ─────────────────────────────── #

async def save_match_v5_collected_players(
    path: str | Path,
    rows: Iterable[Sequence[Any]],
) -> None:
    rows = list(rows)
    if not rows:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    csv_buf = io.StringIO(newline="")
    csv.writer(csv_buf).writerows(rows)
    csv_bytes = csv_buf.getvalue().encode("utf-8")

    compressor = zstd.ZstdCompressor(level=15)
    compressed = compressor.compress(csv_bytes)

    async with aiofiles.open(p, "ab") as f:
        await f.write(compressed)


async def load_match_v5_collected_players(
    path: str | Path,
) -> AsyncIterator[List[Any]]:
    p = Path(path)

    async with aiofiles.open(p, "rb") as f:
        compressed = await f.read()

    decompressed = zstd.ZstdDecompressor().decompress(compressed)
    txt = io.StringIO(decompressed.decode("utf-8"))
    reader = csv.reader(txt)

    for row in reader:
        yield row

# ───────────────────────────────  match_v5_ids  ─────────────────────────────── #

async def save_match_v5_data():
    pass



async def load_match_v5_data():
    pass


# ────────────────────────────────  registry  ─────────────────────────────── #

storages: Mapping[
    str,
    Mapping[str, Mapping[str, Callable[..., Any]]]
] = {
    "league_v4": {
        "default": {"save": save_league_v4, "load": load_league_v4},
    },
    "match_v5": {
        "ids": {"save": save_match_v5_ids, "load": load_match_v5_ids},
        "collected": {
                "save": save_match_v5_collected_players, 
                "load": load_match_v5_collected_players
            }
    },
}
