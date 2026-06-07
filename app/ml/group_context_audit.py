"""Group-level, empirical-Bayes-denoised, debiased context audit.

Re-cuts the champion-specific context audit onto deterministic semantic build/role
groups so per-bin sample sizes are large, then measures the model gap against an
empirical-Bayes-shrunk target with a sampling-variance-debiased Gap MSE.

Why this exists: the champion-specific `HGNN_CONTEXT_EXAMPLES_AUDIT.md` measures the
gap against each split's *raw* empirical win rate, whose per-bin variance `p(1-p)/n`
is irreducible. With median bin n~500 that floor is ~10.5 pp^2, so no architecture
can drive that metric to 0. Pooling identities into semantic groups (large n) and
shrinking the target with empirical Bayes lowers the floor by 5-20x, and subtracting
the residual sampling variance debiases the estimator. The resulting "systematic
Gap MSE" reflects genuine model error / distribution drift and *can* approach 0.

References: multicalibration (Hebert-Johnson et al. 2018), debiased calibration
error (Kumar et al. 2019; Roelofs et al. 2022), James-Stein / empirical Bayes
shrinkage (Efron & Morris 1975).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.ml.context_audit_specs import (
    AuditSpec,
    eb_shrink_targets,
    group_audit_specs,
)
from app.ml.context_examples_audit import AuditData, AuditRow, evaluate_specs
from app.ml.dataset import SPLIT_ORDER

DEFAULT_CONTEXT_CACHE_DIR = Path("app/ml/data/cache")
DEFAULT_PREDICTION_CACHE = Path(
    "app/ml/data/experiments/semantic_architecture_compact_w10_freeze_seed4/"
    "convex_encoder_mix_seed4/audit_focus_side_probability.npy"
)


@dataclass(frozen=True)
class GroupBin:
    label: str
    n: int
    wins: float
    empirical_wr: float
    hgnn_wr: float
    eb_target: float
    sampling_var: float  # pp^2, variance of raw empirical estimate
    eb_var: float        # pp^2, residual variance of the EB target estimate


@dataclass(frozen=True)
class GroupRow:
    spec: AuditSpec
    bins: tuple[GroupBin, ...]


def _empirical_bayes_row(bins: list[tuple[str, int, float, float]]) -> list[GroupBin]:
    """Gaussian empirical-Bayes shrinkage of bin win rates toward the row mean.

    `bins` items are (label, n, empirical_wr, hgnn_wr). Shrinks each bin toward the
    n-weighted row mean by its sampling variance; tau^2 (between-bin variance) is
    estimated by method of moments. Returns GroupBin with EB target and the residual
    variance of that target (the debiasing term).
    """
    pop = [(lab, n, e, h) for (lab, n, e, h) in bins if n > 0]
    if not pop:
        return []
    n_arr = np.array([n for _, n, _, _ in pop], dtype=np.float64)
    p_arr = np.array([e for _, _, e, _ in pop], dtype=np.float64)
    s2 = p_arr * (1.0 - p_arr) / n_arr                  # per-bin sampling variance
    eb, eb_var = eb_shrink_targets(n_arr, p_arr)
    out: list[GroupBin] = []
    for i, (lab, n, e, h) in enumerate(pop):
        out.append(
            GroupBin(
                label=lab,
                n=int(n),
                wins=float(e * n),
                empirical_wr=float(e),
                hgnn_wr=float(h),
                eb_target=float(eb[i]),
                sampling_var=float(s2[i] * 1e4),
                eb_var=float(eb_var[i] * 1e4),
            )
        )
    return out


def build_group_rows(data: AuditData, specs) -> list[GroupRow]:
    raw_rows: tuple[AuditRow, ...] = evaluate_specs(data, specs)
    rows: list[GroupRow] = []
    for r in raw_rows:
        eb = _empirical_bayes_row(
            [(b.label, b.n, b.empirical_wr, b.hgnn_wr) for b in r.bins]
        )
        rows.append(GroupRow(spec=r.spec, bins=tuple(eb)))
    return rows


def _flat(rows: list[GroupRow]) -> list[GroupBin]:
    return [b for r in rows for b in r.bins]


def summarize(rows: list[GroupRow]) -> dict[str, float]:
    bins = _flat(rows)
    if not bins:
        return {}
    raw_gap = np.array([(b.hgnn_wr - b.empirical_wr) * 100 for b in bins])
    eb_gap = np.array([(b.hgnn_wr - b.eb_target) * 100 for b in bins])
    s_var = np.array([b.sampling_var for b in bins])
    eb_var = np.array([b.eb_var for b in bins])
    return {
        "n_bins": len(bins),
        "median_n": float(np.median([b.n for b in bins])),
        "min_n": int(min(b.n for b in bins)),
        # vs RAW per-split empirical WR
        "raw_gap_mse": float(np.mean(raw_gap**2)),
        "raw_floor": float(np.mean(s_var)),
        # vs EB-denoised target
        "eb_gap_mse": float(np.mean(eb_gap**2)),
        "eb_floor": float(np.mean(eb_var)),
        # debiased systematic component (subtract the EB target's own variance)
        "systematic_gap_mse": float(np.mean(eb_gap**2) - np.mean(eb_var)),
        "systematic_gap_mse_clipped": float(np.mean(np.maximum(0.0, eb_gap**2 - eb_var))),
        "eb_mean_abs_gap": float(np.mean(np.abs(eb_gap))),
        "eb_max_abs_gap": float(np.max(np.abs(eb_gap))),
    }


def drift_decomposition(
    *, cache_dir: Path, blue_probability: np.ndarray, specs
) -> dict[str, float]:
    """Compare EB targets across train vs test to isolate distribution drift.

    Returns mean squared (test EB target - train EB target) in pp^2, the part of the
    test gap that is real win-rate movement between the time-ordered splits rather
    than model error.
    """
    train = {
        (r.spec.title, b.label): b.eb_target
        for r in build_group_rows(
            AuditData(context_cache_dir=cache_dir, blue_probability=blue_probability, audit_split="train"),
            specs,
        )
        for b in r.bins
    }
    test_rows = build_group_rows(
        AuditData(context_cache_dir=cache_dir, blue_probability=blue_probability, audit_split="test"),
        specs,
    )
    drifts = [
        (b.eb_target - train[(r.spec.title, b.label)]) * 100
        for r in test_rows
        for b in r.bins
        if (r.spec.title, b.label) in train
    ]
    arr = np.array(drifts)
    return {
        "drift_mse": float(np.mean(arr**2)),
        "drift_mean_abs": float(np.mean(np.abs(arr))),
        "drift_max_abs": float(np.max(np.abs(arr))),
        "n": int(arr.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--context-cache-dir", type=Path, default=DEFAULT_CONTEXT_CACHE_DIR)
    ap.add_argument("--prediction-cache", type=Path, default=DEFAULT_PREDICTION_CACHE)
    ap.add_argument("--per-row", action="store_true", help="print per-row EB gap detail")
    ap.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional machine-readable summary path for experiment reports.",
    )
    args = ap.parse_args()

    blue_probability = np.load(args.prediction_cache, mmap_mode="r")
    specs = group_audit_specs()
    payload: dict[str, object] = {
        "prediction_cache": str(args.prediction_cache),
        "context_cache_dir": str(args.context_cache_dir),
        "n_groups": len(specs),
        "splits": {},
    }
    print(f"prediction cache: {args.prediction_cache}")
    print(f"groups: {len(specs)}\n")

    hdr = f"{'split':<6} {'bins':>5} {'medN':>6} {'minN':>5} | {'raw_mse':>8} {'raw_flr':>8} | {'eb_mse':>7} {'eb_flr':>7} | {'systematic':>10} {'clip':>6} | {'mean|g|':>7} {'max|g|':>7}"
    print(hdr)
    print("-" * len(hdr))
    for split in SPLIT_ORDER:
        data = AuditData(
            context_cache_dir=args.context_cache_dir,
            blue_probability=blue_probability,
            audit_split=split,
        )
        rows = build_group_rows(data, specs)
        s = summarize(rows)
        payload["splits"][split] = s
        if not s:
            print(f"{split:<6} (no populated bins)")
            continue
        print(
            f"{split:<6} {s['n_bins']:>5} {s['median_n']:>6.0f} {s['min_n']:>5} | "
            f"{s['raw_gap_mse']:>8.2f} {s['raw_floor']:>8.2f} | "
            f"{s['eb_gap_mse']:>7.2f} {s['eb_floor']:>7.2f} | "
            f"{s['systematic_gap_mse']:>10.2f} {s['systematic_gap_mse_clipped']:>6.2f} | "
            f"{s['eb_mean_abs_gap']:>7.2f} {s['eb_max_abs_gap']:>7.2f}"
        )
        if args.per_row and split == "val":
            print("\n  per-row EB gap (val), sorted by |systematic| desc:")
            detail = []
            for r in rows:
                for b in r.bins:
                    eb_gap = (b.hgnn_wr - b.eb_target) * 100
                    detail.append(
                        (
                            eb_gap**2 - b.eb_var,
                            r.spec.title,
                            b.label,
                            b.n,
                            b.empirical_wr,
                            b.eb_target,
                            b.hgnn_wr,
                            eb_gap,
                        )
                    )
            top_rows = sorted(detail, reverse=True)[:15]
            payload["val_top_systematic_rows"] = [
                {
                    "systematic_gap_mse": float(sysv),
                    "title": title,
                    "label": lab,
                    "n": int(n),
                    "empirical_wr": float(e),
                    "eb_target": float(eb),
                    "hgnn_wr": float(h),
                    "eb_gap_pp": float(g),
                }
                for sysv, title, lab, n, e, eb, h, g in top_rows
            ]
            for sysv, title, lab, n, e, eb, h, g in top_rows:
                print(f"    {sysv:7.2f}  {title[:42]:42} [{lab:>10}] n={n:>7} emp={e*100:5.1f} eb={eb*100:5.1f} hgnn={h*100:5.1f} gap={g:+5.1f}")

    print()
    drift = drift_decomposition(
        cache_dir=args.context_cache_dir, blue_probability=blue_probability, specs=specs
    )
    payload["drift"] = drift
    print(
        f"train->test drift (EB target movement): mse={drift['drift_mse']:.2f} pp^2  "
        f"mean|d|={drift['drift_mean_abs']:.2f}  max|d|={drift['drift_max_abs']:.2f}  (n={drift['n']})"
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
