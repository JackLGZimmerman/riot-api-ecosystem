"""Rank candidate features by how cleanly they separate two champion sets.

For each candidate feature, prints the standardized mean gap
`(mu_b - mu_a) / pooled_sd` between champion set A and set B on the smoothed,
per-identity values. Use it to find the axis that splits two archetypes that
share another signature.

Candidate features may be any raw `ALL_METRICS` name or any
`DERIVED_METRIC_FUNCS` name; gold-normalised ratios usually separate build
archetypes better than raw volumes because they control for farm.

Example:
    uv run python -m app.classification.embeddings.inspection.discriminate \
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
    EmbeddingConfig,
    IdentityType,
)
from app.classification.embeddings.matrices import _resolve_feature_values
from app.classification.embeddings.report import _load_champion_names
from app.classification.embeddings.tune import load_raw_cached
from app.core.utils.smoothing import apply_hierarchical_shrinkage


def _feature_values(columns: dict, feature: str) -> np.ndarray:
    idx = np.arange(columns["championid"].shape[0], dtype=np.int64)

    class _Rows:
        pass

    rows = _Rows()
    rows.columns = columns
    values = _resolve_feature_values(rows, (feature,), idx)[0]
    return values.astype(np.float64)


def rank_features(
    *,
    set_a: list[str],
    set_b: list[str],
    features: list[str],
) -> None:
    smoothed = apply_hierarchical_shrinkage(load_raw_cached(), EmbeddingConfig())
    columns = smoothed[IdentityType.BASELINE].columns
    names = _load_champion_names()
    inv = {v: k for k, v in names.items()}

    champ = columns["championid"].astype(int)
    ids_a = [inv[n] for n in set_a if n in inv]
    ids_b = [inv[n] for n in set_b if n in inv]
    mask_a = np.isin(champ, ids_a)
    mask_b = np.isin(champ, ids_b)

    print(
        f"set_a n={mask_a.sum()}  set_b n={mask_b.sum()}  "
        "(|gap| desc; gap>0 means set_b higher)"
    )
    rows = []
    for feature in features:
        values = _feature_values(columns, feature)
        a = values[mask_a]
        b = values[mask_b]
        pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-9
        gap = (b.mean() - a.mean()) / pooled
        rows.append((abs(gap), gap, feature, a.mean(), b.mean()))
    for _, gap, feature, am, bm in sorted(rows, reverse=True):
        print(f"{feature:<40} gap={gap:+.3f}  a={am:10.3f}  b={bm:10.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--set-a", nargs="+", required=True)
    parser.add_argument("--set-b", nargs="+", required=True)
    parser.add_argument("--features", nargs="+", required=True)
    args = parser.parse_args()
    rank_features(
        set_a=args.set_a,
        set_b=args.set_b,
        features=args.features,
    )


if __name__ == "__main__":
    main()
