"""Promote trained HGNN seed checkpoints to the production ensemble artifact.

Computes per-seed logits over the cache, fits a bias-only logit calibration on
the train split (default; ``--calibration affine`` also fits a scale) — the
bias restores the blue-side prior that team-swap augmentation suppresses —
then evaluates the calibrated ensemble on test and writes the artifact.

Run from the repo root:
  python -m app.ml.promote --checkpoints seed4.pt seed5.pt ... seed9.pt
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from app.core.logging.logger import setup_logging_config
from app.ml.config import DEFAULT_PRODUCTION_MODEL_PATH, DatasetConfig
from app.ml.dataset import identity_meta, load_splits
from app.ml.hgnn_model import (
    HGNNWinModel,
    load_hgnn_model,
    resolve_device,
    save_hgnn_ensemble,
)
from app.ml.train import (
    _build_sidecar_gatherer,
    _cache_raw_tensor_split,
    _drop_unused_model_arrays,
    _model_uses_sidecar,
    _predict_hgnn_logits,
)

setup_logging_config()
logger = logging.getLogger(__name__)


def _fit_calibration(
    logits: np.ndarray, labels: np.ndarray, device: str, *, fit_scale: bool
) -> tuple[float, float]:
    """Fit logistic ``sigmoid(scale * logit + bias)`` on train rows.

    With ``fit_scale=False`` the scale stays fixed at 1.0 and only the bias is
    fitted (the blue-side prior that team-swap augmentation suppresses). A
    train-fitted scale is in-sample-optimistic — seed logits are sharper on
    rows the members trained on — and measurably overshoots on test, so
    bias-only is the production default (see EXPERIMENTS.md, 2026-06-12).
    """
    x = torch.as_tensor(logits, dtype=torch.float64, device=device)
    y = torch.as_tensor(labels, dtype=torch.float64, device=device)
    scale = torch.ones(1, dtype=torch.float64, device=device, requires_grad=fit_scale)
    bias = torch.zeros(1, dtype=torch.float64, device=device, requires_grad=True)
    params = [scale, bias] if fit_scale else [bias]
    optimizer = torch.optim.LBFGS(params, max_iter=100)

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            scale * x + bias, y
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(scale.item()), float(bias.item())


def _metrics(probs: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probs, 1e-12, 1 - 1e-12)
    return {
        "accuracy": float(np.mean((probs >= 0.5) == (labels > 0.5))),
        "nll": float(
            -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
        ),
    }


def promote(
    checkpoints: list[Path],
    out: Path,
    *,
    batch_size: int = 16384,
    device: str = "auto",
    calibration: str = "bias",
) -> dict[str, float]:
    if calibration not in ("bias", "affine"):
        raise ValueError(f"unknown calibration mode: {calibration!r}")
    device = resolve_device(device)
    dataset_cfg = DatasetConfig()
    meta = identity_meta(dataset_cfg)
    splits = load_splits(
        dataset_cfg, require_counts=True, load_semantic_group_features=True
    )

    members: list[HGNNWinModel] = []
    strength = None
    logits: dict[str, list[np.ndarray]] = {"train": [], "test": []}
    for ckpt in checkpoints:
        model, model_config, ckpt_strength = load_hgnn_model(ckpt, device=device)
        if not isinstance(model, HGNNWinModel):
            raise ValueError(f"promotion expects single-seed checkpoints: {ckpt}")
        if strength is None:
            strength = ckpt_strength
        elif strength != ckpt_strength:
            raise ValueError("seed checkpoints disagree on confidence_strength")
        gatherer = (
            _build_sidecar_gatherer(dataset_cfg, meta, model_config, device=device)
            if _model_uses_sidecar(model_config)
            else None
        )
        for name in ("train", "test"):
            split = _drop_unused_model_arrays(splits[name], model_config)
            raw = _cache_raw_tensor_split(name, split, device="cpu")
            logits[name].append(
                _predict_hgnn_logits(
                    model,
                    raw,
                    batch_size=batch_size,
                    strength=strength,
                    device=device,
                    gatherer=gatherer,
                )
            )
        members.append(model.cpu())
        logger.info("Scored %s on train+test", ckpt.name)

    labels = {name: splits[name].blue_win.astype(np.float64) for name in logits}
    mean = {name: np.mean(values, axis=0) for name, values in logits.items()}
    scale, bias = _fit_calibration(
        mean["train"], labels["train"], device, fit_scale=calibration == "affine"
    )
    test_probs = 1.0 / (1.0 + np.exp(-(scale * mean["test"] + bias)))
    metrics = {
        "n_members": float(len(members)),
        "logit_scale": scale,
        "logit_bias": bias,
        **{f"test_{k}": v for k, v in _metrics(test_probs, labels["test"]).items()},
    }
    logger.info("Calibrated ensemble: %s", metrics)

    assert strength is not None
    save_hgnn_ensemble(
        out,
        members,
        confidence_strength=strength,
        logit_scale=scale,
        logit_bias=bias,
        metrics=metrics,
    )
    logger.info("Wrote production ensemble artifact: %s", out)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_PRODUCTION_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration", choices=("bias", "affine"), default="bias")
    args = parser.parse_args()
    promote(
        args.checkpoints,
        args.out,
        batch_size=args.batch_size,
        device=args.device,
        calibration=args.calibration,
    )


if __name__ == "__main__":
    main()
