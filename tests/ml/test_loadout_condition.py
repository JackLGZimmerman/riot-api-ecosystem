"""Tests for app.ml.loadout_condition (CPU-only, no ClickHouse)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We test internal helpers directly by importing them.
from app.ml.loadout_condition import (
    _sha1_bytes,
    _slot_query,
)


# ---------------------------------------------------------------------------
# Unit helpers
# ---------------------------------------------------------------------------


def _sha1(arr: np.ndarray) -> str:
    return hashlib.sha1(arr.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# _slot_query — pivot row assembly ordering
# ---------------------------------------------------------------------------


def test_slot_query_contains_split_filter() -> None:
    q = _slot_query("train", "", 100)
    assert "split = 'train'" in q


def test_slot_query_keyset_pagination() -> None:
    q = _slot_query("test", "abc123", 500)
    assert "matchid > 'abc123'" in q
    assert "LIMIT 500" in q


def test_slot_query_slot_order() -> None:
    """Slots 0-4 are blue (teamid=100) TOP..UTILITY; slots 5-9 red (teamid=200)."""
    q = _slot_query("train", "", 10)
    positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
    for s, pos in enumerate(positions):
        # Blue slot (teamid=100)
        assert f"champ_{s}" in q
        assert f"ks_{s}" in q
        assert f"teamid = 100 AND ps.teamposition = '{pos}'" in q
    for s, pos in enumerate(positions):
        # Red slot (teamid=200), slots 5-9
        assert f"champ_{s + 5}" in q
        assert f"ks_{s + 5}" in q
        assert f"teamid = 200 AND ps.teamposition = '{pos}'" in q


def test_slot_query_left_join_for_missing_rune() -> None:
    q = _slot_query("train", "", 10)
    # Must use LEFT JOIN so missing rune rows yield 0.
    assert "LEFT JOIN game_data.participant_perk_ids" in q


# ---------------------------------------------------------------------------
# Champion mismatch → raises ValueError
# ---------------------------------------------------------------------------


def _make_split_data(champ_arr: np.ndarray) -> object:
    return SimpleNamespace(champion_id=champ_arr)


def test_alignment_validation_raises_on_champion_mismatch(tmp_path: Path) -> None:
    """build_slot_keystones must raise ValueError if champion arrays disagree."""
    import app.ml.loadout_condition as lc

    n_games = 4
    # Cache champion array
    cache_champ = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]] * n_games, dtype=np.int32)
    # Mismatched array (different champion in slot 0)
    wrong_champ = cache_champ.copy()
    wrong_champ[0, 0] = 99

    # Stub _load_cache_champion_id to return the cache value.
    def fake_load_cache(split: str, cfg: object) -> np.ndarray:
        return cache_champ

    orig_dir = lc._CONDITION_DIR
    lc._CONDITION_DIR = tmp_path

    # Stub the ClickHouse client to return mismatched champion rows.
    mock_client = MagicMock()
    mock_client.query.side_effect = [
        # count query
        MagicMock(result_rows=[(n_games,)]),
        # chunk query: one row per game, matchid then champ_0..9 then ks_0..9
        MagicMock(
            result_rows=[
                ("match_" + str(i),)
                + tuple(int(wrong_champ[i, s]) for s in range(10))
                + (0,) * 10
                for i in range(n_games)
            ]
        ),
    ]

    try:
        with (
            patch("database.clickhouse.client.get_client", return_value=mock_client),
            patch.object(lc, "_load_cache_champion_id", fake_load_cache),
        ):
            cfg = SimpleNamespace(
                cache_dir=tmp_path, player_pivot_table=lc.PLAYER_PIVOT_TABLE
            )
            with pytest.raises(ValueError, match="champion_id mismatch"):
                lc.build_slot_keystones("train", cfg)  # type: ignore[arg-type]
    finally:
        lc._CONDITION_DIR = orig_dir


# ---------------------------------------------------------------------------
# SHA1 staleness check raises
# ---------------------------------------------------------------------------


def test_load_slot_keystones_raises_on_stale_sha1(tmp_path: Path) -> None:
    import app.ml.loadout_condition as lc

    orig_dir = lc._CONDITION_DIR
    lc._CONDITION_DIR = tmp_path

    try:
        n_games = 2
        ks_arr = np.zeros((n_games, 10), dtype=np.int32)
        np.save(str(tmp_path / "test_keystone.npy"), ks_arr)
        # Write meta with wrong sha1
        (tmp_path / "test_keystone_meta.json").write_text(
            json.dumps({"split": "test", "n_games": n_games, "champion_sha1": "deadbeef"}),
            encoding="utf-8",
        )

        current_champ = np.ones((n_games, 10), dtype=np.int32)

        def fake_load_cache(split: str, cfg: object) -> np.ndarray:
            return current_champ

        cfg = SimpleNamespace(cache_dir=tmp_path)
        with (
            patch.object(lc, "_load_cache_champion_id", fake_load_cache),
        ):
            with pytest.raises(ValueError, match="stale"):
                lc.load_slot_keystones("test", cfg)  # type: ignore[arg-type]
    finally:
        lc._CONDITION_DIR = orig_dir


def test_load_slot_keystones_succeeds_with_matching_sha1(tmp_path: Path) -> None:
    import app.ml.loadout_condition as lc

    orig_dir = lc._CONDITION_DIR
    lc._CONDITION_DIR = tmp_path

    try:
        n_games = 3
        ks_arr = np.arange(30, dtype=np.int32).reshape(n_games, 10)
        np.save(str(tmp_path / "test_keystone.npy"), ks_arr)

        current_champ = np.ones((n_games, 10), dtype=np.int32)
        sha1 = _sha1(current_champ)
        (tmp_path / "test_keystone_meta.json").write_text(
            json.dumps({"split": "test", "n_games": n_games, "champion_sha1": sha1}),
            encoding="utf-8",
        )

        def fake_load_cache(split: str, cfg: object) -> np.ndarray:
            return current_champ

        cfg = SimpleNamespace(cache_dir=tmp_path)
        with patch.object(lc, "_load_cache_champion_id", fake_load_cache):
            loaded = lc.load_slot_keystones("test", cfg)  # type: ignore[arg-type]
        assert np.array_equal(loaded, ks_arr)
    finally:
        lc._CONDITION_DIR = orig_dir


# ---------------------------------------------------------------------------
# _sha1_bytes
# ---------------------------------------------------------------------------


def test_sha1_bytes_is_deterministic() -> None:
    arr = np.array([1, 2, 3], dtype=np.int32)
    assert _sha1_bytes(arr) == _sha1_bytes(arr)
    other = np.array([1, 2, 4], dtype=np.int32)
    assert _sha1_bytes(arr) != _sha1_bytes(other)


# ---------------------------------------------------------------------------
# Keystone counts save/load roundtrip
# ---------------------------------------------------------------------------


def test_keystone_counts_roundtrip(tmp_path: Path) -> None:
    """Counts survive the flat disk format and reload bucketed per cell."""
    import app.ml.loadout_condition as lc

    counts = {
        (1, "TOP", 8000): {"a": 60, "b": 15},
        (2, "JUNGLE", 8100): {"c": 7},
    }
    orig_dir, orig_path = lc._CONDITION_DIR, lc.KEYSTONE_COUNTS_PATH
    lc._CONDITION_DIR = tmp_path
    lc.KEYSTONE_COUNTS_PATH = tmp_path / "keystone_counts.json"
    try:
        lc._save_keystone_counts(counts)
        assert lc.load_keystone_counts() == counts
    finally:
        lc._CONDITION_DIR, lc.KEYSTONE_COUNTS_PATH = orig_dir, orig_path
