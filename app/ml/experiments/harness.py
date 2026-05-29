# pyright: reportPrivateImportUsage=false
"""Reusable experiment harness for the structured win-rate puzzle.

Loads the cache once, caches raw tensors on the device, builds structured
features (with optional negative-control shuffles), trains an arbitrary
StructuredModelConfig, and reports the full metric suite plus a
validation-selected decision threshold.

Not a production module. Import `Harness` and call `.run(...)`.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from app.ml.config import DatasetConfig
from app.ml.dataset import SplitData, load_splits
from app.ml.structured_model import (
    DeltaBaselineMode,
    StructuredModelConfig,
    StructuredWinModel,
    resolve_device,
)
from app.ml.train import (
    RawTensorSplit,
    _batch_indices,
    _cache_raw_tensor_split,
    _structured_tensors_from_raw,
    _binary_auc,
    _nll,
    _brier,
    _entropy,
)
from app.ml.utils.calibration import expected_calibration_error

ShuffleSpec = Literal["none", "synergy", "matchup", "all", "base"]


def _ece_adaptive(scores: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    n = scores.size
    if n == 0:
        return float("nan")
    order = np.argsort(scores)
    s, t = scores[order], targets[order]
    total = 0.0
    for idx in np.array_split(np.arange(n), n_bins):
        if idx.size:
            total += idx.size * abs(s[idx].mean() - t[idx].mean())
    return float(total / n)


def select_threshold(scores: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Return (threshold, accuracy) maximizing accuracy on this split."""
    grid = np.linspace(0.30, 0.70, 401)
    preds = scores[None, :] >= grid[:, None]
    acc = (preds == (targets[None, :] > 0.5)).mean(axis=1)
    best = int(np.argmax(acc))
    return float(grid[best]), float(acc[best])


def metric_suite(scores: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    return {
        "n": int(targets.size),
        "acc": float(np.mean((scores >= 0.5) == (targets > 0.5))),
        "auc": _binary_auc(scores, targets),
        "nll": _nll(scores, targets),
        "brier": _brier(scores, targets),
        "ece": expected_calibration_error(scores, targets),
        "adaptive_ece": _ece_adaptive(scores, targets),
        "entropy": _entropy(scores),
    }


@dataclass
class FeatureSet:
    base_features: torch.Tensor
    synergy_objects: torch.Tensor
    matchup_objects: torch.Tensor
    confidence_summaries: torch.Tensor
    role_pair_ids: torch.Tensor
    blue_win: torch.Tensor

    def batch(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "base_features": self.base_features.index_select(0, idx),
            "synergy_objects": self.synergy_objects.index_select(0, idx),
            "matchup_objects": self.matchup_objects.index_select(0, idx),
            "confidence_summaries": self.confidence_summaries.index_select(0, idx),
            "role_pair_ids": self.role_pair_ids,
        }

    def all_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "base_features": self.base_features,
            "synergy_objects": self.synergy_objects,
            "matchup_objects": self.matchup_objects,
            "confidence_summaries": self.confidence_summaries,
            "role_pair_ids": self.role_pair_ids,
        }

    def subset(self, idx: torch.Tensor) -> "FeatureSet":
        return FeatureSet(
            base_features=self.base_features.index_select(0, idx),
            synergy_objects=self.synergy_objects.index_select(0, idx),
            matchup_objects=self.matchup_objects.index_select(0, idx),
            confidence_summaries=self.confidence_summaries.index_select(0, idx),
            role_pair_ids=self.role_pair_ids,
            blue_win=self.blue_win.index_select(0, idx),
        )


class Harness:
    def __init__(
        self,
        *,
        prior_strength: float = 20.0,
        device: str = "auto",
    ) -> None:
        self.device = resolve_device(device)
        self.prior_strength = prior_strength
        self.dataset_cfg = DatasetConfig()
        t0 = time.monotonic()
        self.splits: dict[str, SplitData] = load_splits(self.dataset_cfg, require_counts=True)
        self.raw: dict[str, RawTensorSplit] = {
            name: _cache_raw_tensor_split(name, self.splits[name], device=self.device)
            for name in ("train", "val", "test")
        }
        print(f"[harness] loaded splits + raw tensors in {time.monotonic()-t0:.1f}s on {self.device}")
        self._feat_cache: dict[tuple, dict[str, FeatureSet]] = {}

    def features(
        self,
        *,
        delta_mode: DeltaBaselineMode = "logit",
        shuffle: ShuffleSpec = "none",
        shuffle_seed: int = 0,
    ) -> dict[str, FeatureSet]:
        key = (delta_mode, shuffle, shuffle_seed)
        if key in self._feat_cache:
            return self._feat_cache[key]
        out: dict[str, FeatureSet] = {}
        for name, raw in self.raw.items():
            t = _structured_tensors_from_raw(
                raw, prior_strength=self.prior_strength, delta_baseline_mode=delta_mode
            )
            syn = t["synergy_objects"]
            mat = t["matchup_objects"]
            base = t["base_features"]
            if shuffle != "none":
                g = torch.Generator(device=self.device).manual_seed(shuffle_seed)
                perm = torch.randperm(syn.shape[0], generator=g, device=self.device)
                if shuffle in ("synergy", "all"):
                    syn = syn.index_select(0, perm)
                if shuffle in ("matchup", "all"):
                    mat = mat.index_select(0, perm)
                if shuffle == "base":
                    base = base.index_select(0, perm)
            out[name] = FeatureSet(
                base_features=base,
                synergy_objects=syn,
                matchup_objects=mat,
                confidence_summaries=t["confidence_summaries"],
                role_pair_ids=t["role_pair_ids"],
                blue_win=raw.blue_win,
            )
        self._feat_cache[key] = out
        return out

    def run(
        self,
        name: str,
        model_config: StructuredModelConfig,
        *,
        delta_mode: DeltaBaselineMode = "logit",
        shuffle: ShuffleSpec = "none",
        batch_size: int = 16384,
        max_epochs: int = 25,
        patience: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        verbose: bool = False,
    ) -> dict:
        feats = self.features(delta_mode=delta_mode, shuffle=shuffle)
        y = {s: self.splits[s].blue_win for s in ("train", "val", "test")}
        return self.fit_eval(
            name,
            model_config,
            ft_train=feats["train"],
            ft_es=feats["val"],
            evals={"train": (feats["train"], y["train"]), "val": (feats["val"], y["val"]),
                   "test": (feats["test"], y["test"])},
            thr_from="val",
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            seed=seed,
            verbose=verbose,
        )

    def fit_eval(
        self,
        name: str,
        model_config: StructuredModelConfig,
        *,
        ft_train: "FeatureSet",
        ft_es: "FeatureSet",
        evals: dict[str, tuple["FeatureSet", np.ndarray]],
        thr_from: str,
        batch_size: int = 16384,
        max_epochs: int = 25,
        patience: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        seed: int = 0,
        verbose: bool = False,
    ) -> dict:
        torch.manual_seed(seed)
        if self.device == "cuda":
            torch.cuda.manual_seed_all(seed)
        model = StructuredWinModel(model_config).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()
        rng = np.random.default_rng(seed)
        n_train = ft_train.blue_win.numel()
        es_y = ft_es.blue_win.detach().cpu().numpy().astype(np.float64)
        best_state = copy.deepcopy(model.state_dict())
        best_val = math.inf
        best_epoch = 0
        stale = 0
        t0 = time.monotonic()
        for epoch in range(1, max_epochs + 1):
            model.train()
            for batch_idx in _batch_indices(n_train, batch_size=batch_size, shuffle=True, rng=rng):
                idx = torch.as_tensor(batch_idx, dtype=torch.long, device=self.device)
                batch = ft_train.batch(idx)
                yb = ft_train.blue_win.index_select(0, idx)
                opt.zero_grad(set_to_none=True)
                logits = model(**batch)["final_logit"]
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
            es_scores = self._predict(model, ft_es, batch_size)
            es_nll = _nll(es_scores, es_y)
            if verbose:
                print(f"  e{epoch} es_nll={es_nll:.4f}")
            if es_nll < best_val - 1e-6:
                best_val, best_epoch = es_nll, epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    break
        model.load_state_dict(best_state)
        scores = {s: self._predict(model, ft, batch_size) for s, (ft, _) in evals.items()}
        result: dict = {
            "name": name,
            "best_epoch": best_epoch,
            "seconds": round(time.monotonic() - t0, 1),
            "params": int(sum(p.numel() for p in model.parameters())),
        }
        for s, (_, yy) in evals.items():
            result[s] = metric_suite(scores[s], yy)
        thr, val_thr_acc = select_threshold(scores[thr_from], evals[thr_from][1])
        result["threshold"] = thr
        result["val_thr_acc"] = val_thr_acc
        if "test" in evals:
            yt = evals["test"][1]
            result["test_thr_acc"] = float(((scores["test"] >= thr) == (yt > 0.5)).mean())
        result["_scores"] = scores
        return result

    def _predict(self, model: StructuredWinModel, feat: FeatureSet, batch_size: int) -> np.ndarray:
        model.eval()
        preds = []
        n = feat.blue_win.numel()
        with torch.no_grad():
            for start in range(0, n, batch_size):
                idx = torch.arange(start, min(start + batch_size, n), device=self.device)
                logits = model(**feat.batch(idx))["final_logit"]
                preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(preds).astype(np.float64)


def fmt(r: dict) -> str:
    def line(split: str) -> str:
        m = r[split]
        return (f"{split:5s} acc={m['acc']:.4f} auc={m['auc']:.4f} nll={m['nll']:.4f} "
                f"brier={m['brier']:.4f} ece={m['ece']:.4f}")
    return (
        f"=== {r['name']}  (best_epoch={r['best_epoch']} params={r['params']} {r['seconds']}s)\n"
        f"  {line('train')}\n  {line('val')}\n  {line('test')}\n"
        f"  thr={r['threshold']:.3f} val_thr_acc={r['val_thr_acc']:.4f} test_thr_acc={r['test_thr_acc']:.4f}"
    )
