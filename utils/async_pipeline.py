# utils/async_pipeline.py
from typing import AsyncGenerator, Callable, Iterable, Any, Awaitable
import asyncio

SENTINEL = object()

async def enqueue(gen: AsyncGenerator[Any, None], q: asyncio.Queue, sentinel=SENTINEL) -> None:
    try:
        async for item in gen:
            await q.put(item)
    finally:
        await q.put(sentinel)

async def consumer_loop(
    q: asyncio.Queue,
    save_func: Callable[[str, Iterable[Any]], Awaitable[None]],
    out_path: str,
    batch_size: int,
    num_producers: int,
    sentinel=SENTINEL,
) -> None:
    buffer: list[Any] = []
    done_seen = 0
    while True:
        item = await q.get()
        if item is sentinel:
            done_seen += 1
            if done_seen == num_producers:
                break
            continue
        buffer.append(item)
        if len(buffer) >= batch_size:
            await save_func(out_path, buffer)
            buffer.clear()
    if buffer:
        await save_func(out_path, buffer)
