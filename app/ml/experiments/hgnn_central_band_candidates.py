from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from database.clickhouse.client import get_client
from app.ml.config import DatasetConfig
from app.ml.dataset import identity_meta, load_splits
from app.ml.hgnn_model import load_hgnn_model
from app.ml.train import (
    _build_sidecar_gatherer,
    _cache_raw_tensor_split,
    _drop_unused_model_arrays,
    _model_uses_sidecar,
    _predict_hgnn_outputs,
)

DEFAULT_CACHE_DIR = Path("app/ml/data/cache")
DEFAULT_MODEL_PATH = Path("app/ml/data/experiments/semantic_focus_reference_w3000_cont6/model.pt")
DEFAULT_METRICS_PATH = Path("app/ml/data/experiments/semantic_focus_reference_w3000_cont6/metrics.json")
DEFAULT_ENCODER_SIDECAR_PATH = Path("app/ml/data/experiments/semantic_identity_sidecar_full.npz")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _sql_str(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _load_matchids(split: str) -> list[str]:
    db_split = "validation" if split == "val" else split
    rows = get_client().query(
        "SELECT matchid "
        "FROM game_data_filtered.ml_game_split "
        f"WHERE split = {_sql_str(db_split)} "
        "ORDER BY matchid"
    ).result_rows
    return [str(row[0]) for row in rows]


def _split_ranges(cache_meta: dict[str, Any]) -> dict[str, tuple[int, int]]:
    ranges = {}
    for name, raw in cache_meta["split_ranges"].items():
        ranges[name] = (int(raw["start"]), int(raw["stop"]))
    return ranges


def build_candidates(
    *,
    cache_dir: Path,
    model_path: Path,
    metrics_path: Path,
    encoder_sidecar_path: Path,
    output_path: Path,
    band: tuple[float, float],
    seed: int,
    batch_number: int,
    batch_size: int,
    prediction_batch_size: int,
    device: str,
) -> dict[str, Any]:
    started = time.monotonic()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model, config, strength = load_hgnn_model(model_path, device=device)
    dataset_cfg = DatasetConfig(
        cache_dir=cache_dir,
        encoder_sidecar_path=encoder_sidecar_path,
    )
    meta = identity_meta(dataset_cfg)
    gatherer = (
        _build_sidecar_gatherer(dataset_cfg, meta, config, device=device)
        if _model_uses_sidecar(config)
        else None
    )
    splits = load_splits(
        dataset_cfg,
        require_counts=True,
        load_semantic_group_features=bool(
            config.use_learned_semantic_moe and config.use_semantic_group_features
        ),
    )
    cache_meta = json.loads((cache_dir / "cache_meta.json").read_text(encoding="utf-8"))
    ranges = _split_ranges(cache_meta)
    matchids_by_split = {split: _load_matchids(split) for split in ("val", "test")}

    summary: dict[str, Any] = {}
    candidates: list[dict[str, Any]] = []
    for split_name in ("val", "test"):
        split = splits[split_name]
        raw = _cache_raw_tensor_split(
            split_name,
            _drop_unused_model_arrays(split, config),
            device=device,
        )
        outputs = _predict_hgnn_outputs(
            model,
            raw,
            batch_size=prediction_batch_size,
            strength=strength,
            device=device,
            gatherer=gatherer,
        )
        pred = _sigmoid(outputs["final_logit"])
        labels = np.asarray(split.blue_win, dtype=np.float64)
        central = (pred >= band[0]) & (pred <= band[1])
        classified_blue = pred >= 0.5
        actual_blue = labels >= 0.5
        misses = central & (classified_blue != actual_blue)
        summary[split_name] = {
            "n": int(labels.size),
            "central_n": int(central.sum()),
            "central_accuracy": float(
                ((classified_blue == actual_blue) & central).sum()
                / max(int(central.sum()), 1)
            ),
            "central_misclassified_n": int(misses.sum()),
            "central_blue_wr": float(labels[central].mean()),
            "central_mean_pred": float(pred[central].mean()),
            "central_mis_mean_pred": (
                float(pred[misses].mean()) if int(misses.sum()) else None
            ),
        }

        split_cache_start = ranges[split_name][0]
        split_matchids = matchids_by_split[split_name]
        semantic = split.semantic_group_features
        for idx in np.flatnonzero(misses):
            win_rate = split.win_rate[idx]
            support = split.p1_cnt[idx]
            row = {
                "split": split_name,
                "split_index": int(idx),
                "cache_index": int(split_cache_start + idx),
                "matchid": split_matchids[int(idx)],
                "pred_blue_win": float(pred[idx]),
                "predicted_side": "blue" if pred[idx] >= 0.5 else "red",
                "actual_winner": "blue" if labels[idx] >= 0.5 else "red",
                "blue_win": int(labels[idx]),
                "base_blue_win": float(_sigmoid(outputs["base_logit"][idx])),
                "context_logit": float(outputs["context_logit"][idx]),
                "final_logit": float(outputs["final_logit"][idx]),
                "blue_p1_mean": float(np.mean(win_rate[:5])),
                "red_p1_mean": float(np.mean(win_rate[5:])),
                "blue_p1_min": float(np.min(win_rate[:5])),
                "red_p1_min": float(np.min(win_rate[5:])),
                "blue_p1_cnt_mean": float(np.mean(support[:5])),
                "red_p1_cnt_mean": float(np.mean(support[5:])),
                "blue_p1_cnt_min": float(np.min(support[:5])),
                "red_p1_cnt_min": float(np.min(support[5:])),
                "champion_id": [int(value) for value in split.champion_id[idx].tolist()],
                "build_id": [int(value) for value in split.build_id[idx].tolist()],
            }
            if semantic is not None:
                row["blue_semantic_mean"] = [
                    float(value) for value in np.mean(semantic[idx, :5, :], axis=0).tolist()
                ]
                row["red_semantic_mean"] = [
                    float(value) for value in np.mean(semantic[idx, 5:, :], axis=0).tolist()
                ]
            candidates.append(row)

        del raw, outputs
        if device == "cuda":
            torch.cuda.empty_cache()

    permutation = np.random.default_rng(seed).permutation(len(candidates))
    start = (batch_number - 1) * batch_size
    stop = start + batch_size
    if start >= len(permutation):
        raise ValueError(
            f"batch {batch_number} with size {batch_size} starts after "
            f"{len(permutation)} candidates"
        )
    chosen = permutation[start:stop]
    batch = []
    for batch_position, candidate_index in enumerate(chosen, start=1):
        row = dict(candidates[int(candidate_index)])
        row["candidate_position"] = int(candidate_index)
        row["batch_position"] = batch_position
        batch.append(row)

    output = {
        "source_metrics": str(metrics_path),
        "source_metrics_summary": {
            "val_accuracy": metrics.get("val_accuracy"),
            "test_accuracy": metrics.get("test_accuracy"),
            "decision_threshold": metrics.get("decision_threshold"),
        },
        "model_path": str(model_path),
        "band": list(band),
        "selection": "holdout central-band misclassifications; reproducible random permutation",
        "permutation_seed": seed,
        "batch_number": batch_number,
        "batch_size": batch_size,
        "summary": summary,
        "candidate_count": len(candidates),
        "batch": batch,
        "elapsed_seconds": time.monotonic() - started,
    }
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--encoder-sidecar-path", type=Path, default=DEFAULT_ENCODER_SIDECAR_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--band-low", type=float, default=0.475)
    parser.add_argument("--band-high", type=float, default=0.525)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--batch-number", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--prediction-batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    output = build_candidates(
        cache_dir=args.cache_dir,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        encoder_sidecar_path=args.encoder_sidecar_path,
        output_path=args.output,
        band=(args.band_low, args.band_high),
        seed=args.seed,
        batch_number=args.batch_number,
        batch_size=args.batch_size,
        prediction_batch_size=args.prediction_batch_size,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "batch_number": output["batch_number"],
                "batch_size": output["batch_size"],
                "candidate_count": output["candidate_count"],
                "batch_rows": len(output["batch"]),
                "summary": output["summary"],
                "elapsed_seconds": output["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
