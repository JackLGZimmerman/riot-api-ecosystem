from __future__ import annotations

import numpy as np
import pytest
import torch

from app.ml.hgnn_model import (
    ContextSetEncoder,
    HGNNConfig,
    HGNNWinModel,
    build_hgnn_inputs,
    swap_hgnn_inputs,
)


SET_ENCODERS = ("mean", "deepsets", "set_transformer", "attention", "summary_stats")


@pytest.mark.parametrize("encoder_type", SET_ENCODERS)
def test_context_set_encoder_is_permutation_invariant(encoder_type: str) -> None:
    torch.manual_seed(0)
    encoder = ContextSetEncoder(
        6,
        encoder_type=encoder_type,
        hidden_dim=12,
        heads=3,
        topk=2,
        quantiles=(0.25, 0.5, 0.75),
        weighted_index=5,
    ).eval()
    team = torch.randn(4, 5, 6)
    perm = torch.tensor([2, 4, 0, 3, 1])
    with torch.no_grad():
        base = encoder(team)
        shuffled = encoder(team[:, perm])
    assert torch.allclose(base, shuffled, atol=1e-6), encoder_type


@pytest.mark.parametrize("encoder_type", SET_ENCODERS)
def test_context_head_preserves_explicit_lane_opponent_slot_effects(encoder_type: str) -> None:
    torch.manual_seed(1)
    cfg = HGNNConfig(
        n_champions=50,
        n_builds=5,
        identity_semantic_dim=0,
        identity_profile_dim=0,
        identity_context_dim=6,
        context_include_relational=False,
        context_set_encoder_type=encoder_type,
    )
    model = HGNNWinModel(cfg).eval()
    with torch.no_grad():
        for p in model.context_head.parameters():
            p.normal_()
    ctx = torch.randn(3, 10, 6)
    support = torch.full((3, 10), 500.0)
    perm = torch.tensor([1, 0, 2, 3, 4])
    ctx_permuted = ctx.clone()
    ctx_permuted[:, 5:] = ctx_permuted[:, 5:][:, perm]
    with torch.no_grad():
        base = model._context_logit(ctx, support)
        changed = model._context_logit(ctx_permuted, support)
    assert not torch.allclose(base, changed, atol=1e-6), encoder_type


def _structural_inputs(batch: int = 8) -> dict[str, torch.Tensor]:
    rng = np.random.default_rng(9)
    return build_hgnn_inputs(
        champion_id=rng.integers(0, 50, (batch, 10)),
        build_id=rng.integers(0, 5, (batch, 10)),
        win_rate=rng.random((batch, 10)).astype("f4"),
        matchup_1v1=rng.random((batch, 25)).astype("f4"),
        synergy_2vx=rng.random((batch, 20)).astype("f4"),
        p1_cnt=rng.integers(0, 200, (batch, 10)).astype("f4"),
        m1v1_cnt=rng.integers(0, 200, (batch, 25)).astype("f4"),
        s2vx_cnt=rng.integers(0, 200, (batch, 20)).astype("f4"),
        strength=30.0,
    )


def test_structural_antisymmetry_full_model_logit_and_probability_sum() -> None:
    torch.manual_seed(2)
    model = HGNNWinModel(
        HGNNConfig(
            n_champions=50,
            n_builds=5,
            identity_semantic_dim=0,
            identity_profile_dim=0,
            identity_context_dim=0,
            structural_antisymmetry=True,
        )
    ).eval()
    inputs = _structural_inputs()
    swapped = swap_hgnn_inputs(inputs)
    with torch.no_grad():
        fwd = model(**inputs)["final_logit"]
        rev = model(**swapped)["final_logit"]
        p_fwd = torch.sigmoid(fwd)
        p_rev = torch.sigmoid(rev)
    assert torch.allclose(rev, -fwd, atol=1e-5)
    assert torch.allclose(p_fwd + p_rev, torch.ones_like(p_fwd), atol=1e-6)
