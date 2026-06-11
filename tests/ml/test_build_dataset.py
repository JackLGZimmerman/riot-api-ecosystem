from __future__ import annotations

import pytest

from app.ml.build_dataset import _array_names, _split_counts, _sql_str
from app.ml.config import DatasetConfig


def test_cache_build_player_prior_arrays_are_opt_in() -> None:
    default_names = set(_array_names(include_player_priors=False))
    player_names = set(_array_names(include_player_priors=True))

    assert "player_rate" not in default_names
    assert "player_champ_rate" not in default_names
    assert "player_rate" in player_names
    assert "win_rate" in default_names


def test_cache_build_rejects_global_max_games_cap() -> None:
    with pytest.raises(ValueError, match="max_games is not supported"):
        _split_counts(DatasetConfig(max_games=10))


def test_sql_string_literals_escape_quotes_and_backslashes() -> None:
    assert _sql_str("a'b\\c") == "'a\\'b\\\\c'"
