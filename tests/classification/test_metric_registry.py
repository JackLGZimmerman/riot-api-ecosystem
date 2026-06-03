"""Phase 1 registry guards: byte-stable catalogue, evidence routing, cache hash."""

from __future__ import annotations

import hashlib

from app.classification.embeddings import config
from app.classification.embeddings import load
from app.classification.embeddings import registry as R


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


def test_prior_query_routes_each_metric_by_evidence() -> None:
    # The prior rollup must denominate each metric by its evidence: rate/final by
    # matchups, per-minute by sum_w_timeplayed. (Replaces the old in-Python
    # derive_prior; the rollup now happens in SQL over sufficient statistics.)
    sql, _ = load._prior_query(config.IdentityType.CHAMPION_ROLE, config.EmbeddingConfig())
    assert "toFloat32(sum(b.sum_win) / sum(b.matchups)) AS win" in sql
    assert "60 * sum(b.sum_kills) / sum(b.sum_w_timeplayed)" in sql
    assert "sum(ifNull(f.sum_final_attackdamage, 0)) / sum(b.matchups)) AS attackdamage" in sql

    # The sibling level rolls up via the shared sibling-build SQL helper.
    sib_sql, _ = load._prior_query(config.IdentityType.SIBLING, config.EmbeddingConfig())
    assert "multiIf" in sib_sql


def test_baseline_query_divides_sufficient_statistics() -> None:
    sql, cols = load._baseline_query(config.EmbeddingConfig())
    assert cols[:6] == (
        "championid", "teamposition", "build", "build_group", "matchups", "sum_w_timeplayed",
    )
    assert "toFloat32(b.sum_win / b.matchups) AS win" in sql
    assert "60 * b.sum_kills / b.sum_w_timeplayed" in sql
