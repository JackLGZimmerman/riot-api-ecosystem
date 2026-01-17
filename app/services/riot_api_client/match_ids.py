import asyncio
from typing import AsyncGenerator, Iterable

from .base import RiotAPI
from .utils import (
    PlayerCrawlState,
    fetch_json_with_carry_over,
)

MAX_PAGE_START = 900
MAX_PAGE_COUNT = 100
MAX_IN_FLIGHT: int = 128


INITIAL_BACKFILL_DAYS = 30


async def stream_match_ids(
    riot_api: RiotAPI,
    *,
    initial_states: Iterable[PlayerCrawlState],
    ts: int,
    max_in_flight: int = MAX_IN_FLIGHT,
) -> AsyncGenerator[list[str], None]:
    work_q: asyncio.Queue[PlayerCrawlState | None] = asyncio.Queue()
    out_q: asyncio.Queue[list[str] | BaseException | None] = asyncio.Queue()

    for st in initial_states:
        work_q.put_nowait(st)

    async def worker() -> None:
        try:
            while True:
                state = await work_q.get()
                try:
                    if state is None:
                        return

                    url = state.base_url.format(
                        start=state.next_page_start,
                        endTime=ts,
                    )

                    new_state, match_ids = await fetch_json_with_carry_over(
                        carry_over=(state,),
                        url=url,
                        location=state.continent,
                        riot_api=riot_api,
                    )

                    await out_q.put(match_ids)

                    if (
                        new_state.next_page_start != MAX_PAGE_START
                        and len(match_ids) == MAX_PAGE_COUNT
                    ):
                        work_q.put_nowait(
                            new_state._replace(
                                next_page_start=new_state.next_page_start
                                + MAX_PAGE_COUNT
                            )
                        )
                finally:
                    work_q.task_done()
        except BaseException as e:
            await out_q.put(e)

    async def closer(workers: list[asyncio.Task[None]]) -> None:
        await work_q.join()

        for _ in workers:
            work_q.put_nowait(None)

        await asyncio.gather(*workers, return_exceptions=True)
        await out_q.put(None)

    workers = [asyncio.create_task(worker()) for _ in range(max_in_flight)]
    close_task = asyncio.create_task(closer(workers))

    try:
        while True:
            item = await out_q.get()

            if item is None:
                return

            if isinstance(item, BaseException):
                raise item

            yield item
    finally:
        for t in workers:
            t.cancel()
        close_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
