"""Rank candidate features by how cleanly they separate two champion sets.

For each candidate feature, prints the standardized mean gap
`(mu_b - mu_a) / pooled_sd` between champion set A and set B on the smoothed,
per-identity values for one phase. Use it to find the axis that splits two
archetypes that share another signature (e.g. burst assassins vs skirmishers,
which share `kills_to_assists` and differ only on a survivability axis).

Candidate features may be any raw `ALL_METRICS` name or any
`DERIVED_METRIC_FUNCS` name; gold-normalised ratios usually separate build
archetypes better than raw volumes because they control for game stage/farm.

Example:
    uv run python -m app.classification.embeddings.inspection.discriminate \
        --phase mid \
        --set-a Pantheon Akshan Varus Zed Talon Kindred \
        --set-b Jax Irelia Viego MasterYi Briar Fiora \
        --features damageselfmitigated_to_goldearned_ratio \
            durability_total_to_goldearned_ratio armor magicresist \
            healthmax_to_goldearned_ratio kills_to_assists_ratio
"""

from __future__ import annotations

import argparse

import numpy as np

from app.classification.embeddings.config import (
    PHASES,
    EmbeddingConfig,
    IdentityType,
)
from app.classification.embeddings.matrices import _resolve_feature_values
from app.classification.embeddings.posteriors import apply_hierarchical_shrinkage
from app.classification.embeddings.report import _load_champion_names
from app.classification.embeddings.tune import load_raw_cached


def _phase_feature_values(
    columns: dict, feature: str, phase_mask: np.ndarray
) -> np.ndarray:
    idx = np.where(phase_mask)[0]

    class _Rows:
        pass

    rows = _Rows()
    rows.columns = columns
    values = _resolve_feature_values(rows, (feature,), idx)[0]
    return values.astype(np.float64)


def rank_features(
    *,
    phase: str,
    set_a: list[str],
    set_b: list[str],
    features: list[str],
) -> None:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    smoothed = apply_hierarchical_shrinkage(load_raw_cached(), EmbeddingConfig())
    columns = smoothed[IdentityType.BASELINE].columns
    names = _load_champion_names()
    inv = {v: k for k, v in names.items()}

    phase_mask = columns["phase"] == phase
    champ = columns["championid"][phase_mask].astype(int)
    ids_a = [inv[n] for n in set_a if n in inv]
    ids_b = [inv[n] for n in set_b if n in inv]
    mask_a = np.isin(champ, ids_a)
    mask_b = np.isin(champ, ids_b)

    print(
        f"phase={phase}  set_a n={mask_a.sum()}  set_b n={mask_b.sum()}  "
        "(|gap| desc; gap>0 means set_b higher)"
    )
    rows = []
    for feature in features:
        values = _phase_feature_values(columns, feature, phase_mask)
        a = values[mask_a]
        b = values[mask_b]
        pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-9
        gap = (b.mean() - a.mean()) / pooled
        rows.append((abs(gap), gap, feature, a.mean(), b.mean()))
    for _, gap, feature, am, bm in sorted(rows, reverse=True):
        print(f"{feature:<40} gap={gap:+.3f}  a={am:10.3f}  b={bm:10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default=PHASES[0])
    parser.add_argument("--set-a", nargs="+", required=True)
    parser.add_argument("--set-b", nargs="+", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    args = parser.parse_args()
    rank_features(
        phase=args.phase,
        set_a=args.set_a,
        set_b=args.set_b,
        features=args.features,
    )


if __name__ == "__main__":
    main()
