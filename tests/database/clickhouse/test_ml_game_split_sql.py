"""Contract checks for the per-patch chronological ml_game_split build SQL."""

from __future__ import annotations

import re
from pathlib import Path

SCHEMA_DIR = Path(__file__).resolve().parents[3] / "database" / "clickhouse" / "schema"
BUILD_SQL = (SCHEMA_DIR / "5900_ml_game_split_build.sql").read_text()


def test_split_build_emits_only_train_and_test_labels() -> None:
    labels = set(re.findall(r"'(\w+)'", BUILD_SQL))
    assert "train" in labels
    assert "test" in labels
    assert "validation" not in BUILD_SQL


def test_split_build_partitions_chronologically_by_patch() -> None:
    assert "PARTITION BY season, patch" in BUILD_SQL
    assert (
        "ORDER BY gamestarttimestamp ASC, gamecreation ASC, matchid ASC" in BUILD_SQL
    )
    assert "floor(patch_games * 0.8)" in BUILD_SQL
