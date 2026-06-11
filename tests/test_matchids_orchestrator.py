from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from app.core.config.constants import Continent, Region
from app.worker.pipelines.matchids_orchestrator import (
    MatchIDCollectorState,
    MatchIDSaver,
    build_initial_player_states,
)
from app.worker.pipelines.orchestrator import OrchestrationContext
from database.clickhouse.operations.players import PlayerKeyRow


def test_build_initial_player_states_uses_full_player_key_for_timestamp() -> None:
    states = build_initial_player_states(
        [
            PlayerKeyRow("same-puuid", "RANKED_SOLO_5x5", "na1"),
            PlayerKeyRow("same-puuid", "RANKED_FLEX_SR", "na1"),
        ],
        {("same-puuid", "RANKED_SOLO_5x5")},
        200,
        ts=300,
        start_time_floor=100,
    )

    solo, flex = states

    assert solo.region == Region.NA1
    assert solo.continent == Continent.AMERICAS
    assert "startTime=200" in solo.base_url
    assert "queue=420" in solo.base_url
    assert "startTime=100" in flex.base_url
    assert "queue=440" in flex.base_url


def test_matchid_saver_rejects_partial_player_crawl(monkeypatch) -> None:
    calls: list[str] = []

    async def insert_matchids_stream_in_batches(items, run_id):
        calls.append("insert_matchids")
        async for _ in items:
            pass

    def record(name):
        def inner(*args, **kwargs):
            calls.append(name)

        return inner

    monkeypatch.setattr(
        "app.worker.pipelines.matchids_orchestrator.insert_matchids_stream_in_batches",
        insert_matchids_stream_in_batches,
    )
    monkeypatch.setattr(
        "app.worker.pipelines.matchids_orchestrator.delete_failed_puuid_timestamp",
        record("delete_failed_puuid_timestamp"),
    )
    monkeypatch.setattr(
        "app.worker.pipelines.matchids_orchestrator.delete_matchid_puuids",
        record("delete_matchid_puuids"),
    )
    monkeypatch.setattr(
        "app.worker.pipelines.matchids_orchestrator.delete_matchids",
        record("delete_matchids"),
    )
    monkeypatch.setattr(
        "app.worker.pipelines.matchids_orchestrator.insert_puuids_in_batches",
        record("insert_puuids"),
    )
    monkeypatch.setattr(
        "app.worker.pipelines.matchids_orchestrator.upsert_puuid_timestamp",
        record("upsert_ts"),
    )
    monkeypatch.setattr(
        "app.worker.pipelines.matchids_orchestrator.delete_old_puuid_timestamps",
        record("delete_old_ts"),
    )

    async def empty_items():
        if False:
            yield []

    state = MatchIDCollectorState(
        initial_states=[],
        full_player_keys=[("puuid-a", "RANKED_SOLO_5x5")],
        failed_player_keys={("puuid-a", "RANKED_SOLO_5x5")},
        ts=123,
    )
    ctx = OrchestrationContext(
        ts=123,
        run_id=UUID("11111111-1111-1111-1111-111111111111"),
        pipeline="match_ids",
    )

    with pytest.raises(RuntimeError, match="Match ID crawl failed"):
        asyncio.run(MatchIDSaver().save(empty_items(), state, ctx))

    assert calls == [
        "insert_matchids",
        "delete_failed_puuid_timestamp",
        "delete_matchid_puuids",
        "delete_matchids",
    ]
