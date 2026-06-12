from __future__ import annotations

import numpy as np
import torch

from app.ml.hgnn_model import (
    HGNNConfig,
    HGNNEnsemble,
    HGNNWinModel,
    load_hgnn_model,
    save_hgnn_ensemble,
)
from app.ml.promote import _fit_calibration


def _toy_inputs(n: int = 4) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {
        "champion_id": torch.randint(0, 10, (n, 10), generator=generator),
        "build_id": torch.randint(0, 3, (n, 10), generator=generator),
        "mu_1vx": torch.full((n, 10), 0.5),
        "var_1vx": torch.full((n, 10), 0.01),
        "conf_1vx": torch.full((n, 10), 0.8),
        "log_count_1vx": torch.full((n, 10), 3.0),
    }


def test_ensemble_round_trip_applies_mean_and_calibration(tmp_path) -> None:
    config = HGNNConfig(n_champions=10, n_builds=3)
    members = [HGNNWinModel(config).eval() for _ in range(2)]
    path = tmp_path / "ensemble.pt"
    save_hgnn_ensemble(
        path,
        members,
        confidence_strength=30.0,
        logit_scale=1.25,
        logit_bias=-0.04,
        metrics={"test_accuracy": 0.58},
    )

    loaded, loaded_config, strength = load_hgnn_model(path)

    assert isinstance(loaded, HGNNEnsemble)
    assert loaded_config.n_champions == 10
    assert strength == 30.0
    assert loaded.logit_scale == 1.25
    assert loaded.logit_bias == -0.04

    inputs = _toy_inputs()
    with torch.no_grad():
        member_logits = torch.stack(
            [member(**inputs)["final_logit"] for member in loaded.members]
        )
        expected = 1.25 * member_logits.mean(dim=0) - 0.04
        actual = loaded(**inputs)["final_logit"]

    torch.testing.assert_close(actual, expected)


def test_fit_calibration_recovers_known_scale_and_bias() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 2.0, size=20000)
    probs = 1.0 / (1.0 + np.exp(-(1.4 * logits - 0.3)))
    labels = (rng.uniform(size=logits.size) < probs).astype(np.float64)

    scale, bias = _fit_calibration(logits, labels, device="cpu", fit_scale=True)

    assert abs(scale - 1.4) < 0.1
    assert abs(bias + 0.3) < 0.1


def test_fit_calibration_bias_only_keeps_unit_scale() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(0.0, 2.0, size=20000)
    probs = 1.0 / (1.0 + np.exp(-(logits - 0.3)))
    labels = (rng.uniform(size=logits.size) < probs).astype(np.float64)

    scale, bias = _fit_calibration(logits, labels, device="cpu", fit_scale=False)

    assert scale == 1.0
    assert abs(bias + 0.3) < 0.1
