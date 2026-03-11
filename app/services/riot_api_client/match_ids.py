from __future__ import annotations

import asyncio
from typing import AsyncIterator, Iterable

from app.services.riot_api_client.base import FetchOutcome, RiotAPI
from app.services.riot_api_client.utils import PlayerCrawlState, spreading

MAX_PAGE_START = 900
MAX_PAGE_COUNT = 100
MAX_IN_FLIGHT = 64

PlayerKey = tuple[str, str]
MatchIDStreamItem = tuple[PlayerKey, list[str], str | None]


class _Done:
    pass


def _player_key(state: PlayerCrawlState) -> PlayerKey:
    return (state.puuid, state.queue_type.value)


def _state_url(state: PlayerCrawlState) -> str:
    return state.base_url.format(start=state.next_page_start)


async def stream_match_ids(
    riot_api: RiotAPI,
    *,
    initial_states: Iterable[PlayerCrawlState],
    max_in_flight: int = MAX_IN_FLIGHT,
) -> AsyncIterator[MatchIDStreamItem]:
    work_q: asyncio.Queue[PlayerCrawlState | None] = asyncio.Queue()
    out_q: asyncio.Queue[MatchIDStreamItem | BaseException | _Done] = asyncio.Queue()

    seed_url_jobs = spreading(
        [(_state_url(state), state) for state in initial_states],
        key_fn=lambda job: job[1].continent,
    )
    for _, state in seed_url_jobs:
        work_q.put_nowait(state)

    async def worker() -> None:
        try:
            while True:
                state = await work_q.get()
                try:
                    if state is None:
                        return

                    page_match_ids = await _fetch_page_match_ids(
                        riot_api=riot_api,
                        state=state,
                        page_start=state.next_page_start,
                    )

                    if page_match_ids:
                        await out_q.put((_player_key(state), page_match_ids, None))

                    if (
                        state.next_page_start < MAX_PAGE_START
                        and len(page_match_ids) == MAX_PAGE_COUNT
                    ):
                        work_q.put_nowait(
                            state._replace(
                                next_page_start=state.next_page_start + MAX_PAGE_COUNT
                            )
                        )
                except Exception as exc:
                    if state is not None:
                        await out_q.put((_player_key(state), [], str(exc)))
                    else:
                        await out_q.put(exc)
                finally:
                    work_q.task_done()
        except Exception as exc:
            await out_q.put(exc)

    async def closer(workers: list[asyncio.Task[None]]) -> None:
        await work_q.join()

        for _ in workers:
            work_q.put_nowait(None)

        await asyncio.gather(*workers, return_exceptions=True)
        await out_q.put(_Done())

    workers = [asyncio.create_task(worker()) for _ in range(max_in_flight)]
    close_task = asyncio.create_task(closer(workers))

    try:
        while True:
            item = await out_q.get()

            if isinstance(item, _Done):
                return

            if isinstance(item, BaseException):
                raise item

            yield item
    finally:
        for task in workers:
            task.cancel()
        close_task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        await asyncio.gather(close_task, return_exceptions=True)


async def _fetch_page_match_ids(
    *,
    riot_api: RiotAPI,
    state: PlayerCrawlState,
    page_start: int,
) -> list[str]:
    url = state.base_url.format(start=page_start)
    result = await riot_api.fetch_json_detailed(
        url=url,
        location=state.continent,
    )

    if result.outcome is not FetchOutcome.OK:
        if result.outcome is FetchOutcome.HTTP_NON_RETRYABLE and result.status == 404:
            return []
        raise RuntimeError(
            "crawl failed "
            f"outcome={result.outcome.value} status={result.status} "
            f"puuid={state.puuid} queue_type={state.queue_type.value} page_start={page_start}"
        )

    payload = result.data
    if not isinstance(payload, list):
        raise RuntimeError(
            "crawl returned unexpected payload "
            f"type={type(payload).__name__} puuid={state.puuid} "
            f"queue_type={state.queue_type.value} page_start={page_start}"
        )

    return [mid for mid in payload if isinstance(mid, str)]
