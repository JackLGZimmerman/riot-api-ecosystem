from __future__ import annotations

import numpy as np

from app.classification.embeddings.config import (
    ALL_METRICS,
    DERIVED_METRIC_FUNCS,
    EmbeddingConfig,
    IdentityType,
    raw_and_derived_metric_names,
    raw_metric_names,
)
from app.classification.embeddings.load import LevelRows, _baseline_query
from app.classification.embeddings.matrices import build_level_matrix


def test_baseline_query_excludes_challenge_data() -> None:
    baseline_query = _baseline_query("train").lower()

    assert "participant_challenges" not in baseline_query
    assert "challenge_" not in baseline_query


def test_preserved_metric_catalogue_contains_raw_and_derived_metrics() -> None:
    raw = raw_metric_names()
    features = raw_and_derived_metric_names()

    assert raw == ALL_METRICS
    assert set(ALL_METRICS).issubset(features)
    assert set(DERIVED_METRIC_FUNCS).issubset(features)
    assert "physicaldamagedealttochampions_share" in features
    assert "matchups" not in features
    assert not any("challenge" in name.lower() for name in features)


def test_default_embedding_config_uses_raw_and_derived_metrics() -> None:
    cfg = EmbeddingConfig()

    assert cfg.feature_set == raw_and_derived_metric_names()


def test_build_level_matrix_materializes_derived_metrics_from_smoothed_sources() -> None:
    rows = _level_rows(
        {
            "smoothed_physicaldamagedealttochampions": [3.0, 0.0],
            "smoothed_totaldamagedealttochampions": [6.0, 0.0],
        }
    )
    cfg = EmbeddingConfig(feature_set=("physicaldamagedealttochampions_share",))

    matrix = build_level_matrix(rows, cfg)

    assert matrix is not None
    assert matrix.feature_names == ("physicaldamagedealttochampions_share",)
    assert matrix.matrix.shape == (2, 1)
    assert np.isfinite(matrix.matrix).all()


def test_build_level_matrix_clips_extreme_standardised_values() -> None:
    rows = _level_rows({"smoothed_kills": [0.0, 0.0, 0.0, 0.0, 1.0e12]})
    cfg = EmbeddingConfig(feature_set=("kills",), matrix_clip_value=3.0)

    matrix = build_level_matrix(rows, cfg)

    assert matrix is not None
    assert np.max(np.abs(matrix.matrix)) <= 3.0


def _level_rows(extra_columns: dict[str, list[float]]) -> LevelRows:
    n = max((len(values) for values in extra_columns.values()), default=2)
    columns: dict[str, np.ndarray] = {
        "championid": np.arange(1, n + 1, dtype=np.int32),
        "teamposition": np.resize(np.asarray(["TOP", "JUNGLE"], dtype=object), n),
        "build": np.resize(np.asarray(["tank", "bruiser"], dtype=object), n),
        "matchups": np.linspace(20.0, 30.0, n, dtype=np.float32),
    }
    columns.update(
        {
            name: np.asarray(values, dtype=np.float32)
            for name, values in extra_columns.items()
        }
    )
    return LevelRows(
        level=IdentityType.BASELINE,
        key_columns=("championid", "teamposition", "build"),
        columns=columns,
        n=n,
    )
