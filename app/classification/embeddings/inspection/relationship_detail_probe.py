"""Probe a relationship-detail artifact for a specific matchup.

Default probe:
    python -m app.classification.embeddings.inspection.relationship_detail_probe
"""

from __future__ import annotations

import argparse

import numpy as np

from app.classification.embeddings.config import RELATIONSHIP_DETAIL_CACHE_DIR


def _find_row(keys: np.ndarray, key: tuple) -> int | None:
    for idx, row in enumerate(keys):
        if tuple(row.tolist() if hasattr(row, "tolist") else row) == key:
            return idx
    return None


def probe_sion_yone() -> None:
    path = RELATIONSHIP_DETAIL_CACHE_DIR / "m1v1.npz"
    if not path.exists():
        raise SystemExit(f"Missing relationship detail artifact: {path}")
    sion = (14, "TOP", "ar_tank")
    yone = (777, "TOP", "crit")
    swapped = sion > yone
    left, right = (yone, sion) if swapped else (sion, yone)
    sign = -1.0 if swapped else 1.0

    with np.load(path, allow_pickle=True) as payload:
        key = (*left, *right)
        idx = _find_row(payload["exact_keys"], key)
        if idx is None:
            raise SystemExit(f"No exact Sion/Yone row found for key={key}")
        features = [str(x) for x in payload["feature_names"]]
        raw = payload["exact_raw_values"][idx].astype(float) * sign
        matchups = float(payload["exact_matchups"][idx])

    print("Sion TOP ar_tank vs Yone TOP crit (Sion perspective)")
    print(f"matchups={matchups:.0f}")
    for name, value in zip(features, raw, strict=True):
        print(f"{name}={value:.4f}")
    print()
    print("Yone perspective is the negative of directional diff/net-rate fields.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    probe_sion_yone()


if __name__ == "__main__":
    main()
