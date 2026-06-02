"""Phase 1 registry guards: byte-stable catalogue, evidence routing, cache hash."""

from __future__ import annotations

import hashlib

import numpy as np

from app.classification.embeddings import config, load
from app.classification.embeddings import registry as R
from app.classification.embeddings.load import LevelRows, derive_prior


def _digest(names: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()[:16]


# Frozen snapshots of the pre-registry ordering. Any reorder/rename of the
# catalogue must update these deliberately.
ALL_METRICS_DIGEST = "f8adad3b51396d98"
RAW_AND_DERIVED_DIGEST = "42ef0702948645aa"


def test_catalogue_ordering_is_byte_stable() -> None:
    assert _digest(R.ALL_METRICS) == ALL_METRICS_DIGEST
    assert _digest(config.raw_and_derived_metric_names()) == RAW_AND_DERIVED_DIGEST


def test_config_reexports_match_registry() -> None:
    assert config.ALL_METRICS is R.ALL_METRICS
    assert config.DERIVED_METRIC_FUNCS is R.DERIVED_METRIC_FUNCS
    assert config.RATE_METRICS is R.RATE_METRICS
    assert config.LARGEST_AVG_METRICS is R.LARGEST_AVG_METRICS
    assert config.FINAL_SNAPSHOT_AVG_METRICS is R.FINAL_SNAPSHOT_AVG_METRICS
    assert config.PER_MINUTE_METRICS is R.PER_MINUTE_METRICS
    assert config.RATE_LIKE_METRICS is R.RATE_LIKE_METRICS


def test_specs_are_unique_and_ordered() -> None:
    names = [spec.name for spec in R.FULL_GAME_SPECS]
    assert len(names) == len(set(names))
    assert tuple(s.name for s in R.RAW_SPECS) == R.ALL_METRICS
    assert tuple(s.name for s in R.DERIVED_SPECS) == tuple(R.DERIVED_METRIC_FUNCS)
    assert len(R.RAW_SPECS) == 66
    assert len(R.DERIVED_SPECS) == 81


def test_no_challenge_metrics() -> None:
    assert not any("challenge" in s.name.lower() for s in R.FULL_GAME_SPECS)


def test_derived_dependencies_resolve_to_raw_metrics() -> None:
    for spec in R.DERIVED_SPECS:
        assert spec.dependencies, f"{spec.name} introspected no dependencies"
        for dep in spec.dependencies:
            assert dep in R.ALL_METRICS, f"{spec.name} depends on unknown {dep}"


def test_evidence_routing() -> None:
    for metric in R.RATE_LIKE_METRICS:
        assert R.EVIDENCE_BY_RAW_METRIC[metric] is R.Evidence.MATCHUPS
    for metric in R.PER_MINUTE_METRICS:
        assert R.EVIDENCE_BY_RAW_METRIC[metric] is R.Evidence.SUM_W_TIMEPLAYED
    assert set(R.EVIDENCE_BY_RAW_METRIC) == set(R.ALL_METRICS)


def test_catalogue_hash_is_deterministic() -> None:
    assert R.catalogue_hash() == R.catalogue_hash()
    assert len(R.catalogue_hash()) == 16


def _synthetic_baseline() -> LevelRows:
    # Two rows, same (champion, role), so they fold into one CHAMPION_ROLE prior.
    # matchups and timeplayed weight the rows oppositely so a rate metric and a
    # per-minute metric resolve to distinct prior values.
    n = 2
    columns: dict[str, np.ndarray] = {
        "championid": np.asarray([1, 1], dtype=np.int32),
        "teamposition": np.asarray(["TOP", "TOP"], dtype=object),
        "build": np.asarray(["a", "b"], dtype=object),
        "build_group": np.asarray(["g", "g"], dtype=object),
        "matchups": np.asarray([1.0, 3.0], dtype=np.float32),
        "sum_w_timeplayed": np.asarray([3.0, 1.0], dtype=np.float64),
    }
    for metric in R.ALL_METRICS:
        columns[metric] = np.zeros(n, dtype=np.float32)
    columns["win"] = np.asarray([1.0, 0.0], dtype=np.float32)  # rate -> matchups
    columns["kills"] = np.asarray([1.0, 0.0], dtype=np.float32)  # per-min -> timeplayed
    return LevelRows(
        config.IdentityType.BASELINE,
        config.LEVEL_KEY[config.IdentityType.BASELINE],
        columns,
        n,
    )


def test_derive_prior_routes_each_metric_by_evidence() -> None:
    prior = derive_prior(config.IdentityType.CHAMPION_ROLE, _synthetic_baseline())
    assert prior.n == 1
    # matchups-weighted: (1*1 + 3*0) / 4
    assert prior.columns["win"][0] == np.float32(0.25)
    # timeplayed-weighted: (3*1 + 1*0) / 4
    assert prior.columns["kills"][0] == np.float32(0.75)


def test_baseline_cache_rejects_stale_catalogue(tmp_path, monkeypatch) -> None:
    path = tmp_path / "baseline.npz"
    rows = LevelRows(
        config.IdentityType.BASELINE,
        config.LEVEL_KEY[config.IdentityType.BASELINE],
        {
            "championid": np.asarray([1], dtype=np.int32),
            "teamposition": np.asarray(["TOP"], dtype=object),
            "build": np.asarray(["a"], dtype=object),
        },
        1,
    )
    load._save_level_rows(path, rows)

    loaded = load._load_level_rows(path, config.IdentityType.BASELINE)
    assert loaded is not None and loaded.n == 1

    monkeypatch.setattr(load, "catalogue_hash", lambda: "deadbeefdeadbeef")
    assert load._load_level_rows(path, config.IdentityType.BASELINE) is None
