from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from app.core.utils.smoothing import nested_shrunk_rate
from app.ml.hgnn_model import HGNNConfig, HGNNWinModel, build_hgnn_inputs, load_hgnn_model, swap_hgnn_inputs
from app.ml.predictor import _interaction_pooling_from_cache_meta
from app.ml.priors import PriorTables


def test_nested_pooling_backs_off_to_dense_parent_effective_support() -> None:
    build_rate = np.array([[0.50]], dtype=np.float64)
    build_count = np.array([[0.0]], dtype=np.float64)
    nobuild_rate = np.array([[0.50]], dtype=np.float64)
    nobuild_count = np.array([[0.0]], dtype=np.float64)
    champ_rate = np.array([[0.70]], dtype=np.float64)
    champ_count = np.array([[800.0]], dtype=np.float64)

    pooled, eff_n = nested_shrunk_rate(
        [build_rate, nobuild_rate, champ_rate],
        [build_count, nobuild_count, champ_count],
        strengths=[20.0, 20.0, 1.0],
        floor_prior=0.5,
        amplification_threshold=0.0,
    )

    assert pooled.shape == (1, 1)
    assert eff_n.shape == (1, 1)
    assert np.isclose(pooled[0, 0], (0.70 * 800.0 + 0.5) / 801.0)
    assert eff_n[0, 0] == 800.0


def test_hgnn_inputs_use_raw_relationship_support_features() -> None:
    inputs = build_hgnn_inputs(
        champion_id=np.zeros((1, 10), dtype=np.int64),
        build_id=np.zeros((1, 10), dtype=np.int64),
        win_rate=np.full((1, 10), 0.5, dtype=np.float32),
        matchup_1v1=np.full((1, 25), 0.7, dtype=np.float32),
        synergy_2vx=np.full((1, 20), 0.6, dtype=np.float32),
        p1_cnt=np.zeros((1, 10), dtype=np.float32),
        m1v1_cnt=np.zeros((1, 25), dtype=np.float32),
        s2vx_cnt=np.zeros((1, 20), dtype=np.float32),
        strength=30.0,
    )

    assert float(inputs["conf_1v1"][0, 0]) == 0.0
    assert float(inputs["log_count_1v1"][0, 0]) == 0.0
    assert float(inputs["missing_1v1"][0, 0]) == 1.0
    assert float(inputs["delta_logit_1v1"][0, 0]) > 0.0
    assert "var_1v1" not in inputs
    assert "var_2vx" not in inputs


def test_direct_relationship_tensor_keeps_blue_win_2vx_signing() -> None:
    model = HGNNWinModel(HGNNConfig(logit_clip=None))
    delta_1v1 = torch.zeros(1, 25)
    delta_2vx = torch.arange(1, 21, dtype=torch.float32).reshape(1, 20)

    signed = model._signed_relationship_tensor(delta_2vx, delta_1v1)

    assert torch.equal(signed[:, 25:35], delta_2vx[:, :10])
    assert torch.equal(signed[:, 35:45], -delta_2vx[:, 10:])


def test_swap_hgnn_inputs_transforms_relationship_support_and_signed_deltas() -> None:
    inputs = {
        "champion_id": torch.arange(10).reshape(1, 10),
        "build_id": torch.arange(10).reshape(1, 10),
        "mu_1vx": torch.linspace(0.1, 1.0, 10).reshape(1, 10),
        "var_1vx": torch.arange(10, dtype=torch.float32).reshape(1, 10),
        "mu_2vx": torch.arange(20, dtype=torch.float32).reshape(1, 20),
        "mu_1v1": torch.full((1, 25), 0.6),
        "conf_1vx": torch.ones(1, 10),
        "log_count_1vx": torch.arange(10, dtype=torch.float32).reshape(1, 10),
        "missing_1vx": torch.zeros(1, 10),
        "conf_2vx": torch.arange(20, dtype=torch.float32).reshape(1, 20),
        "log_count_2vx": torch.arange(20, dtype=torch.float32).reshape(1, 20) + 300,
        "missing_2vx": torch.zeros(1, 20),
        "conf_1v1": torch.arange(25, dtype=torch.float32).reshape(1, 25),
        "log_count_1v1": torch.arange(25, dtype=torch.float32).reshape(1, 25) + 400,
        "missing_1v1": torch.zeros(1, 25),
        "delta_logit_2vx": torch.arange(20, dtype=torch.float32).reshape(1, 20),
        "delta_logit_1v1": torch.arange(25, dtype=torch.float32).reshape(1, 25),
        "identity_semantic": torch.arange(10 * 4, dtype=torch.float32).reshape(1, 10, 4),
        "m1v1_detail": torch.arange(25 * 3, dtype=torch.float32).reshape(1, 25, 3),
        "s2vx_detail": torch.arange(20 * 2, dtype=torch.float32).reshape(1, 20, 2),
    }

    swapped = swap_hgnn_inputs(inputs)

    assert torch.equal(swapped["log_count_2vx"], torch.cat([inputs["log_count_2vx"][:, 10:], inputs["log_count_2vx"][:, :10]], dim=1))
    assert torch.equal(
        swapped["conf_1v1"],
        inputs["conf_1v1"].reshape(1, 5, 5).transpose(1, 2).reshape(1, 25),
    )
    assert torch.equal(
        swapped["delta_logit_1v1"],
        -inputs["delta_logit_1v1"].reshape(1, 5, 5).transpose(1, 2).reshape(1, 25),
    )
    assert torch.equal(
        swapped["identity_semantic"],
        torch.cat([inputs["identity_semantic"][:, 5:], inputs["identity_semantic"][:, :5]], dim=1),
    )
    assert torch.equal(
        swapped["m1v1_detail"],
        -inputs["m1v1_detail"].reshape(1, 5, 5, 3).transpose(1, 2).reshape(1, 25, 3),
    )
    assert torch.equal(
        swapped["s2vx_detail"],
        torch.cat([inputs["s2vx_detail"][:, 10:], inputs["s2vx_detail"][:, :10]], dim=1),
    )


def _tiny_inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    return build_hgnn_inputs(
        champion_id=np.zeros((batch, 10), dtype=np.int64),
        build_id=np.zeros((batch, 10), dtype=np.int64),
        win_rate=np.full((batch, 10), 0.5, dtype=np.float32),
        matchup_1v1=np.full((batch, 25), 0.52, dtype=np.float32),
        synergy_2vx=np.full((batch, 20), 0.51, dtype=np.float32),
        p1_cnt=np.full((batch, 10), 10.0, dtype=np.float32),
        m1v1_cnt=np.full((batch, 25), 5.0, dtype=np.float32),
        s2vx_cnt=np.full((batch, 20), 5.0, dtype=np.float32),
        strength=30.0,
        identity_semantic=np.zeros((batch, 10, 64), dtype=np.float32),
        m1v1_detail=np.zeros((batch, 25, 16), dtype=np.float32),
        s2vx_detail=np.zeros((batch, 20, 16), dtype=np.float32),
    )


def _tiny_config(**overrides: object) -> HGNNConfig:
    values = {
        "n_champions": 4,
        "n_builds": 2,
        "node_dim": 16,
        "edge_hidden": 8,
        "value_hidden": (8,),
        "gate_hidden": (4,),
        "node_init_hidden": (16,),
        "readout_hidden": (16,),
        "residual_head_hidden": (16,),
    }
    values.update(overrides)
    return HGNNConfig(**values)


def test_direct_hgnn_forward_returns_finite_logits() -> None:
    model = HGNNWinModel(_tiny_config())

    out = model(**_tiny_inputs())["final_logit"]

    assert out.shape == (2,)
    assert torch.isfinite(out).all()


def test_legacy_hgnn_payload_ignores_removed_config_keys(tmp_path: Path) -> None:
    model = HGNNWinModel(_tiny_config())
    path = tmp_path / "legacy.pt"
    config = {
        **model.config.__dict__,
        "removed_experimental_key": 1,
    }
    torch.save(
        {
            "model_type": "hgnn",
            "model_config": config,
            "confidence_strength": 30.0,
            "state_dict": model.state_dict(),
        },
        path,
    )

    loaded, loaded_config, _ = load_hgnn_model(path)

    assert loaded_config.node_dim == model.config.node_dim
    assert not hasattr(loaded, "removed_experimental_key")


def test_prior_table_backoff_lookups_match_training_orientation() -> None:
    priors = PriorTables(
        p1={},
        m1v1={(1, "TOP", "carry", 2, "JUNGLE", "tank"): (0.65, 10)},
        s2vx={(1, "TOP", "carry", 3, "MIDDLE", "mage"): (0.58, 12)},
        m1v1_nb={(1, "TOP", 2, "JUNGLE"): (0.62, 100)},
        m1v1_champ={(1, 2): (0.60, 800)},
        s2vx_nb={(1, "TOP", 3, "MIDDLE"): (0.57, 200)},
        s2vx_bg={},
        s2vx_champ={(1, 3): (0.56, 900)},
    )

    blue = [(1, "TOP", "carry")]
    red = [(2, "JUNGLE", "tank")]
    assert priors.lookup_1v1_blue(blue, red)[0][0] == 0.65
    assert priors.lookup_1v1_blue_nobuild(blue, red)[0][0] == 0.62
    assert priors.lookup_1v1_blue_champ(blue, red)[0][0] == 0.60

    team = [
        (3, "MIDDLE", "mage"),
        (1, "TOP", "carry"),
        (9, "JUNGLE", "none"),
        (10, "BOTTOM", "none"),
        (11, "UTILITY", "none"),
    ]
    s2vx_wr, s2vx_cnt = priors.lookup_2vx_team(team)
    s2vx_nb_wr, _ = priors.lookup_2vx_team_nobuild(team)
    s2vx_ch_wr, _ = priors.lookup_2vx_team_champ(team)
    assert s2vx_wr[0] == 0.58
    assert s2vx_cnt[0] == 12
    assert s2vx_nb_wr[0] == 0.57
    assert s2vx_ch_wr[0] == 0.56


def test_predictor_reuses_cache_meta_interaction_strengths(tmp_path: Path) -> None:
    strengths = {
        "m1v1": [53.2, 149.1, 276.0],
        "s2vx": [54.1, 191.0, 292.9],
    }
    (tmp_path / "cache_meta.json").write_text(
        json.dumps(
            {
                "smoothing": {
                    "interaction_nested_pooling": True,
                    "interaction_level_strengths": strengths,
                }
            }
        )
    )

    nested_pooling, level_strengths, s2vx_ladder = _interaction_pooling_from_cache_meta(
        tmp_path,
        fallback_strength=20.0,
    )

    assert nested_pooling is True
    assert level_strengths == strengths
    assert s2vx_ladder == ("build", "nobuild", "champion")


def test_predictor_falls_back_when_cache_meta_lacks_complete_strengths(tmp_path: Path) -> None:
    (tmp_path / "cache_meta.json").write_text(
        json.dumps(
            {
                "smoothing": {
                    "interaction_nested_pooling": True,
                    "interaction_level_strengths": {"m1v1": [53.2], "s2vx": [54.1, 191.0, 292.9]},
                }
            }
        )
    )

    nested_pooling, level_strengths, s2vx_ladder = _interaction_pooling_from_cache_meta(
        tmp_path,
        fallback_strength=20.0,
    )

    assert nested_pooling is False
    assert level_strengths == {"m1v1": [20.0], "s2vx": [20.0]}
    assert s2vx_ladder == ("build", "nobuild", "champion")
