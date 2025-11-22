from collections.abc import AsyncIterable, AsyncIterator
from typing import TypeVar

T = TypeVar("T")


async def deduplicate_by_puuid(
    rows: AsyncIterable[T],
    seen_puuids: set[str],
    new_puuids: set[str],
) -> AsyncIterator[T]:
    async for row in rows:
        puuid = getattr(row, "puuid", None)
        if puuid is None:
            continue

        if puuid in seen_puuids:
            continue

        seen_puuids.add(puuid)
        new_puuids.add(puuid)
        yield row