"""Materialised-build SQL: DDL coverage, hygiene, staged shard->combine shapes."""

from __future__ import annotations

import pytest

from app.classification.embeddings import build_tables as B
from app.classification.embeddings import temporal as T
from app.classification.embeddings.config import FINAL_SNAPSHOT_AVG_METRICS


def test_ddl_lists_every_metric() -> None:
    identity = B._identity_ddl()
    assert B.IDENTITY_BASE in identity
    assert "matchups UInt64" in identity
    assert "sum_w_timeplayed Float64" in identity
    for metric in B.IDENTITY_SUM_METRICS:
        assert f"sum_{metric} Float64" in identity
    final = B._final_ddl(B.FINAL_BASE)
    for metric in FINAL_SNAPSHOT_AVG_METRICS:
        assert f"sum_final_{metric} Float64" in final
    bins = B._bins_ddl()
    assert "frames UInt64" in bins
    for metric in T.TEMPORAL_METRICS:
        assert f"sum_{metric} Float64" in bins
    for metric in T.EVENT_METRICS:
        assert f"ev_{metric} Float64" in bins


def test_identity_build_is_clean() -> None:
    sql = B._identity_insert("train")
    assert "challenge" not in sql.lower()
    assert "count() AS matchups" not in sql  # stored as toUInt64(count())
    assert "toUInt64(count()) AS matchups" in sql
    assert "s.split = 'train'" in sql
    # nullable per-minute metric is coalesced before summing
    assert "coalesce(pc.damagedealttoepicmonsters, 0)" in sql


def test_temporal_build_is_sharded_and_clean() -> None:
    bounds = tuple(f"M{i}" for i in range(B.K_SHARDS - 1))  # K-1 interior cut points
    stat = B._temporal_stat_insert("train", 3, bounds)
    assert "challenge" not in stat.lower()
    # matchid-range shard (prunable on the leading ORDER BY key), not cityHash64.
    assert "cityHash64(t.matchid)" not in stat
    assert "t.matchid >= 'M2' AND t.matchid < 'M3'" in stat
    assert "ps.matchid >= 'M2' AND ps.matchid < 'M3'" in stat  # CTE aligned to same range
    assert f"least(intDiv(t.frame_timestamp, {T.FRAME_MS}), {T.N_BUCKETS - 1})" in stat
    assert "count() AS frames" in stat
    ev = B._temporal_ev_insert("train", 3, bounds)
    assert "matchid >= 'M2' AND matchid < 'M3'" in ev  # event scans pruned too
    assert T.CHAMPION_KILL_TABLE in ev
    assert T.TURRET_PLATE_TABLE in ev
    assert "arrayJoin(assistingparticipantids)" in ev
    for lane in ("TOP_LANE", "MID_LANE", "BOT_LANE"):
        assert f"lanetype = '{lane}'" in ev
    # open-ended first/last shards omit the missing bound
    assert "t.matchid < 'M0'" in B._temporal_stat_insert("train", 0, bounds)
    assert f"t.matchid >= 'M{B.K_SHARDS - 2}'" in B._temporal_stat_insert(
        "train", B.K_SHARDS - 1, bounds
    )
    # combine sums the per-shard partials back to one row per (identity, bucket)
    combine = B._temporal_combine("train")
    assert "sum(frames) AS frames" in combine


def test_context_combine_keeps_separate_counts() -> None:
    sql = B._context_combine("train")
    assert "sum(cnt) AS cnt_team" in sql
    assert "sum(cnt) AS cnt_matchup" in sql
    assert "challenge" not in sql.lower()


def test_assert_built_guards_missing_and_stale(monkeypatch) -> None:
    class _Result:
        def __init__(self, rows):
            self.result_rows = rows

    class _Client:
        def __init__(self, rows):
            self._rows = rows

        def query(self, _sql):
            return _Result(self._rows)

    monkeypatch.setattr(B, "get_client", lambda: _Client([]))
    with pytest.raises(RuntimeError, match="not built"):
        B.assert_built("abc")

    monkeypatch.setattr(B, "get_client", lambda: _Client([("stalehash",)]))
    with pytest.raises(RuntimeError, match="stale catalogue"):
        B.assert_built("abc")

    monkeypatch.setattr(B, "get_client", lambda: _Client([("abc",)]))
    B.assert_built("abc")  # matching hash -> no raise
