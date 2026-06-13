# pyright: reportPrivateImportUsage=false

"""Cache-side marginalised evaluation for the build-intent plan (Phase A).

Scores the production checkpoint over the v32 cache with hypothesised
pregame build worlds instead of observed labels: per game, the train-only
catalog supplies one prior per slot, the joint worlds are enumerated
best-first, every retained world is scored in batched forward passes, and
output probabilities are averaged with the unnormalised joint weights
divided by retained mass.

Accepted modes (``marginal``, ``modal``) never read a held-out row's
observed ``build_id``; ``oracle`` reproduces the recorded production path
and is diagnostics-only.

Run from the repo root, e.g.:
  python -m app.ml.marginal_eval --mode marginal --worlds 512 --split test
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.core.logging.logger import setup_logging_config
from app.ml.build_catalog import (
    BUILD_SOURCE_ORACLE_OBSERVED,
    BUILD_SOURCE_PREGAME_MARGINAL,
    BuildCatalog,
    build_catalog,
    enumerate_joint_worlds,
    validate_accepted_build_source,
)
from app.ml.config import (
    DEFAULT_PRODUCTION_MODEL_PATH,
    ML_DATA_DIR,
    POSITIONS,
    DatasetConfig,
)
from app.ml.dataset import SplitData, identity_meta, load_splits
from app.ml.hgnn_model import (
    HGNNEnsemble,
    HGNNWinModel,
    build_hgnn_inputs,
    load_hgnn_model,
    model_requires_semantic_group_features,
    model_uses_encoder_sidecar,
    resolve_device,
)
from app.ml.pregame import HypothesisTables, build_hypothesis_tables
from app.ml.priors import load_priors
from app.ml.promote import _fit_calibration
from app.ml.semantic_group_features import (
    build_semantic_group_features,
    static_hp_range_lookups,
)
from app.ml.train import (
    _build_sidecar_gatherer,
    _cache_raw_tensor_split,
    _drop_unused_model_arrays,
    _predict_hgnn_logits,
)
setup_logging_config()
logger = logging.getLogger(__name__)

EPS = 1e-12
DEFAULT_OUT_DIR = ML_DATA_DIR / "experiments"


@dataclass(frozen=True)
class MarginalScores:
    probabilities: np.ndarray
    labels: np.ndarray
    retained_mass: np.ndarray
    n_worlds: np.ndarray
    fallback_counts: dict[str, int]


def _metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    p = np.clip(probabilities.astype(np.float64), EPS, 1.0 - EPS)
    y = labels.astype(np.float64)
    bins = np.minimum((p * 15).astype(np.int64), 14)
    ece = 0.0
    for b in range(15):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return {
        "n": int(y.size),
        "accuracy": float(np.mean((p >= 0.5) == (y > 0.5))),
        "nll": float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": float(ece),
    }


def _logit(p: np.ndarray) -> np.ndarray:
    clipped = np.clip(p.astype(np.float64), EPS, 1.0 - EPS)
    return np.log(clipped / (1.0 - clipped))


def score_split_marginal(
    model: HGNNWinModel | HGNNEnsemble,
    split: SplitData,
    catalog: BuildCatalog,
    tables: HypothesisTables,
    *,
    strength: float,
    device: str,
    gatherer,
    k_slot: int,
    max_worlds: int,
    early_stop_mass: float,
    batch_rows: int = 16384,
    max_games: int | None = None,
    log_every: int = 50_000,
) -> MarginalScores:
    """Marginal probabilities for one split without reading its build_id."""
    champion_id = split.champion_id
    if champion_id is None:
        raise ValueError("cache is missing champion_id")
    n_games = champion_id.shape[0] if max_games is None else min(
        max_games, champion_id.shape[0]
    )
    needs_semantic = model_requires_semantic_group_features(model.config)
    hp_lookup, range_lookup = static_hp_range_lookups() if needs_semantic else (None, None)
    n_champions = tables.win_rate.shape[0] - 1
    slot_roles = np.arange(10) % 5

    # Per-(champ, role) candidate cache.
    candidate_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, str]] = {}

    def candidates(champ: int, role_idx: int) -> tuple[np.ndarray, np.ndarray, str]:
        key = (champ, role_idx)
        cached = candidate_cache.get(key)
        if cached is None:
            vector = catalog.prior_vector(champ, POSITIONS[role_idx])
            cached = (
                np.asarray(vector.hgnn_build_ids, dtype=np.int64),
                np.asarray(vector.probabilities, dtype=np.float64),
                vector.fallback_source,
            )
            candidate_cache[key] = cached
        return cached

    probabilities = np.zeros(n_games, dtype=np.float64)
    retained = np.zeros(n_games, dtype=np.float64)
    world_counts = np.zeros(n_games, dtype=np.int64)
    fallback_counts: dict[str, int] = {}

    pending_games: list[int] = []  # game index per pending world row
    pending_builds: list[np.ndarray] = []  # [10] build ids per world
    pending_weights: list[float] = []

    def flush() -> None:
        if not pending_games:
            return
        game_idx = np.asarray(pending_games, dtype=np.int64)
        build_id = np.stack(pending_builds)
        weights = np.asarray(pending_weights, dtype=np.float64)
        champ = np.asarray(champion_id[:n_games][game_idx], dtype=np.int64)
        # Out-of-range ids map to the reserve row (n_champions), matching the
        # runtime predictor — never to a real champion via clipping.
        champ = np.where((champ < 0) | (champ >= n_champions), n_champions, champ)
        win_rate = tables.win_rate[champ, slot_roles, build_id]
        p1_cnt = tables.p1_cnt[champ, slot_roles, build_id]
        semantic = None
        if needs_semantic:
            semantic = build_semantic_group_features(
                context_raw=tables.context[champ, slot_roles, build_id],
                champion_id=champ,
                build_id=build_id,
                build_vocab=model.config.build_vocab,
                hp_lookup=hp_lookup,
                range_lookup=range_lookup,
            )
        sidecar = (
            gatherer.gather(
                torch.as_tensor(champ, dtype=torch.long, device=device),
                torch.as_tensor(build_id, dtype=torch.long, device=device),
            )
            if gatherer is not None
            else {}
        )
        inputs = build_hgnn_inputs(
            champion_id=champ,
            build_id=build_id,
            win_rate=win_rate,
            p1_cnt=p1_cnt,
            strength=strength,
            semantic_group_features=semantic,
            device=device,
            **sidecar,
        )
        with torch.no_grad():
            logits = model(**inputs)["final_logit"]
            world_probs = (
                torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
            )
        np.add.at(probabilities, game_idx, world_probs * weights)
        pending_games.clear()
        pending_builds.clear()
        pending_weights.clear()

    started = time.monotonic()
    for game in range(n_games):
        champ_row = np.asarray(champion_id[game], dtype=np.int64)
        slot_candidates = [
            candidates(int(champ_row[s]), int(slot_roles[s])) for s in range(10)
        ]
        for _, _, source in slot_candidates:
            fallback_counts[source] = fallback_counts.get(source, 0) + 1
        selections, weights, mass = enumerate_joint_worlds(
            [probs for _, probs, _ in slot_candidates],
            k_slot=k_slot,
            max_worlds=max_worlds,
            early_stop_mass=early_stop_mass,
        )
        retained[game] = mass
        world_counts[game] = weights.size
        builds_by_slot = [ids for ids, _, _ in slot_candidates]
        for w in range(selections.shape[0]):
            pending_games.append(game)
            pending_builds.append(
                np.array(
                    [builds_by_slot[s][selections[w, s]] for s in range(10)],
                    dtype=np.int64,
                )
            )
            pending_weights.append(float(weights[w]))
        if len(pending_games) >= batch_rows:
            flush()
        if log_every and (game + 1) % log_every == 0:
            logger.info(
                "scored %d/%d games (%.1f games/s)",
                game + 1,
                n_games,
                (game + 1) / max(time.monotonic() - started, EPS),
            )
    flush()
    probabilities /= np.maximum(retained, EPS)
    return MarginalScores(
        probabilities=probabilities,
        labels=np.asarray(split.blue_win[:n_games], dtype=np.float64),
        retained_mass=retained,
        n_worlds=world_counts,
        fallback_counts=fallback_counts,
    )


def _mass_report(scores: MarginalScores, mass_floor: float) -> dict[str, float]:
    mass = scores.retained_mass
    return {
        "mean": float(mass.mean()),
        "p10": float(np.percentile(mass, 10)),
        "p50": float(np.percentile(mass, 50)),
        "min": float(mass.min()),
        "share_below_floor": float(np.mean(mass < mass_floor)),
        "mass_floor": float(mass_floor),
        "mean_worlds": float(scores.n_worlds.mean()),
    }


def run(
    *,
    mode: str,
    worlds: int,
    k_slot: int,
    early_stop_mass: float,
    mass_floor: float,
    split_name: str,
    calibrate: bool,
    calibration_max_games: int | None,
    max_games: int | None,
    model_path: Path,
    out: Path | None,
    device: str = "auto",
) -> dict:
    device = resolve_device(device)
    dataset_cfg = DatasetConfig()
    meta = identity_meta(dataset_cfg)
    model, model_config, strength = load_hgnn_model(model_path, device=device)
    model.eval()
    # Only oracle scores the cached observed-build rows (and their cached
    # semantic features); accepted modes rebuild every feature per world.
    splits = load_splits(
        dataset_cfg,
        require_counts=True,
        load_semantic_group_features=(mode == "oracle"),
    )
    gatherer = (
        _build_sidecar_gatherer(dataset_cfg, meta, model_config, device=device)
        if model_uses_encoder_sidecar(model_config)
        else None
    )

    payload: dict = {
        "mode": mode,
        "split": split_name,
        "model_path": str(model_path),
        "device": device,
    }
    if mode == "oracle":
        # Diagnostics only: exactly the production cache path (observed builds).
        payload["build_source"] = BUILD_SOURCE_ORACLE_OBSERVED
        split = _drop_unused_model_arrays(splits[split_name], model_config)
        raw = _cache_raw_tensor_split(split_name, split, device="cpu")
        logits = _predict_hgnn_logits(
            model,
            raw,
            batch_size=16384,
            strength=strength,
            device=device,
            gatherer=gatherer,
        )
        n = logits.size if max_games is None else min(max_games, logits.size)
        probabilities = 1.0 / (1.0 + np.exp(-logits[:n]))
        payload["metrics"] = _metrics(
            probabilities, np.asarray(split.blue_win[:n], dtype=np.float64)
        )
        return _finish(payload, out)

    payload["build_source"] = validate_accepted_build_source(
        BUILD_SOURCE_PREGAME_MARGINAL
    )
    if mode == "modal":
        worlds, k_slot = 1, 1
    payload.update(
        {
            "worlds": worlds,
            "k_slot": k_slot,
            "early_stop_mass": early_stop_mass,
        }
    )
    priors = load_priors()
    catalog = build_catalog(priors.p1, tuple(model_config.build_vocab))
    catalog.assert_pregame_native()
    payload["catalog_version"] = catalog.version
    tables = build_hypothesis_tables(
        dataset_cfg,
        priors,
        n_champions=int(model_config.n_champions),
        build_vocab=tuple(model_config.build_vocab),
    )

    scores = score_split_marginal(
        model,
        splits[split_name],
        catalog,
        tables,
        strength=strength,
        device=device,
        gatherer=gatherer,
        k_slot=k_slot,
        max_worlds=worlds,
        early_stop_mass=early_stop_mass,
        max_games=max_games,
    )
    payload["metrics"] = _metrics(scores.probabilities, scores.labels)
    payload["retained_joint_mass"] = _mass_report(scores, mass_floor)
    payload["fallback_slot_counts"] = scores.fallback_counts
    if out is not None:
        # Persist per-game scores before calibration so failures there cost
        # nothing and calibration variants can be probed offline.
        np.savez_compressed(
            out.with_suffix(".npz"),
            probabilities=scores.probabilities,
            labels=scores.labels,
            retained_mass=scores.retained_mass,
        )

    if calibrate:
        # Fresh bias-only calibration on train rows scored by the same
        # procedure (source pregame_marginal_build); the production scale/bias
        # was fitted under observed-build conditioning. A train-fitted scale is
        # in-sample-optimistic (see EXPERIMENTS.md, 2026-06-12), so the scale
        # stays at 1.0. Train rows here see full-count self-inclusive priors
        # (the training cache used LOO), so the fitted bias is slightly
        # optimistic; uncalibrated test metrics are unaffected.
        train_scores = score_split_marginal(
            model,
            splits["train"],
            catalog,
            tables,
            strength=strength,
            device=device,
            gatherer=gatherer,
            k_slot=k_slot,
            max_worlds=worlds,
            early_stop_mass=early_stop_mass,
            max_games=calibration_max_games,
        )
        if out is not None:
            np.savez_compressed(
                out.with_name(out.stem + "_calibration_rows.npz"),
                probabilities=train_scores.probabilities,
                labels=train_scores.labels,
            )
        scale, bias = _fit_calibration(
            _logit(train_scores.probabilities),
            train_scores.labels,
            device,
            fit_scale=False,
        )
        calibrated = 1.0 / (
            1.0 + np.exp(-(scale * _logit(scores.probabilities) + bias))
        )
        payload["marginal_calibration"] = {
            "scale": scale,
            "bias": bias,
            "fit_games": int(train_scores.labels.size),
        }
        payload["calibrated_metrics"] = _metrics(calibrated, scores.labels)
    return _finish(payload, out)


def _finish(payload: dict, out: Path | None) -> dict:
    logger.info("result: %s", json.dumps(payload, indent=2, sort_keys=True))
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        logger.info("wrote %s", out)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("marginal", "modal", "oracle"), required=True)
    parser.add_argument("--worlds", type=int, default=512)
    parser.add_argument("--k-slot", type=int, default=3)
    parser.add_argument("--early-stop-mass", type=float, default=0.90)
    parser.add_argument("--mass-floor", type=float, default=0.35)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--calibrate", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--calibration-max-games", type=int, default=200_000)
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_PRODUCTION_MODEL_PATH,
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    out = args.out
    if out is None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = DEFAULT_OUT_DIR / f"marginal_eval_{args.mode}_{args.split}_{stamp}.json"
    run(
        mode=args.mode,
        worlds=args.worlds,
        k_slot=args.k_slot,
        early_stop_mass=args.early_stop_mass,
        mass_floor=args.mass_floor,
        split_name=args.split,
        calibrate=args.calibrate and args.mode != "oracle",
        calibration_max_games=args.calibration_max_games,
        max_games=args.max_games,
        model_path=args.model_path,
        out=out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
