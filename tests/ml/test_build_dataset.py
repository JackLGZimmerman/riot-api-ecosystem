from __future__ import annotations

import pytest

from app.ml.build_dataset import _split_counts, _sql_str
from app.ml.config import DatasetConfig


def test_cache_build_rejects_global_max_games_cap() -> None:
    with pytest.raises(ValueError, match="max_games is not supported"):
        _split_counts(DatasetConfig(max_games=10))


def test_sql_string_literals_escape_quotes_and_backslashes() -> None:
    assert _sql_str("a'b\\c") == "'a\\'b\\\\c'"
