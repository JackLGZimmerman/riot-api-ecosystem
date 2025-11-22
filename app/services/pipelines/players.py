# app/services/pipeline.py
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from app.core.config import settings
from app.models import BasicBoundsConfig, EliteBoundsConfig
from app.services.riot_api_client.league_v4 import (
    stream_elite_players,
    stream_sub_elite_players,
)
from app.services.utils.deduplicate import deduplicate_by_puuid
from app.services.utils.file import get_file_operations, zstandard_streamed_export_async

FILE_OPERATIONS: dict[str, Callable] = get_file_operations()


async def run_player_collection_pipeline(
    elite_bounds: EliteBoundsConfig,
    sub_elite_bounds: BasicBoundsConfig,
) -> None:
    base_data_dir: Path = settings.data_path

    index_dir = base_data_dir / "indexes"
    snapshots_dir = base_data_dir / "snapshots"
    index_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "seen_puuids.txt"

    # ============ Saving: load existing index ============
    load_index = FILE_OPERATIONS["load_puuid_index"]
    append_index = FILE_OPERATIONS["append_puuid_index"]

    seen_puuids: set[str] = load_index(index_path)
    new_puuids: set[str] = set()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ============ Producer + Consumer: elite snapshot ============
    elite_output = snapshots_dir / f"elite_{ts}.zst"

    # Producer: stream -> dedupe
    elite_rows = deduplicate_by_puuid(
        rows=stream_elite_players(elite_bounds),
        seen_puuids=seen_puuids,
        new_puuids=new_puuids,
    )

    # Consumer: export to compressed file
    await zstandard_streamed_export_async(
        rows=elite_rows,
        path=elite_output,
    )

    # ============ Producer + Consumer: sub-elite snapshot ============
    sub_elite_output = snapshots_dir / f"sub_elite_{ts}.zst"

    sub_rows = deduplicate_by_puuid(
        rows=stream_sub_elite_players(sub_elite_bounds),
        seen_puuids=seen_puuids,
        new_puuids=new_puuids,
    )

    await zstandard_streamed_export_async(
        rows=sub_rows,
        path=sub_elite_output,
    )

    # ============ Saving: append new puuids to index ============
    if new_puuids:
        append_index(index_path, new_puuids)
