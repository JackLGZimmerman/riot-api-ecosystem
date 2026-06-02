"""Unit tests for the low-rank identity-conditioned context head.

Covers the acceptance-critical invariants: zero-init opt-in, blue/red
antisymmetry under team swap, per-player support gating, slot/lane-opponent
alignment, and missing-identity behaviour. The atlas-leakage / lookup
correctness checks live in tests/classification/test_identity_context.py.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from app.ml.hgnn_model import (
    HGNNConfig,
    HGNNWinModel,
    build_hgnn_inputs,
    swap_hgnn_inputs,
)

RAW = 62
DENSE_CTX = 24  # 14 interpretable + 10 PCA tail


def _model(*, source: str = "raw", residual_mlp: bool = False) -> HGNNWinModel:
    cfg = HGNNConfig(
        n_champions=50,
        n_builds=5,
        identity_profile_dim=0,
        identity_context_dim=DENSE_CTX,
        context_interpretable_dim=14,
        identity_context_raw_dim=RAW,
        use_identity_conditioned_context=True,
        identity_context_conditioning_type="low_rank",
        identity_context_source=source,
        identity_context_rank=8,
        identity_context_hidden_dim=32,
        identity_context_emb_dim=16,
        identity_context_use_residual_mlp=residual_mlp,
    )
    return HGNNWinModel(cfg).eval()


def _inputs(batch: int = 8, seed: int = 0) -> dict:
    r = np.random.default_rng(seed)
    return build_hgnn_inputs(
        champion_id=r.integers(0, 50, (batch, 10)),
        build_id=r.integers(0, 5, (batch, 10)),
        win_rate=r.random((batch, 10)).astype("f4"),
        matchup_1v1=r.random((batch, 25)).astype("f4"),
        synergy_2vx=r.random((batch, 20)).astype("f4"),
        p1_cnt=r.integers(0, 200, (batch, 10)).astype("f4"),
        m1v1_cnt=r.integers(0, 200, (batch, 25)).astype("f4"),
        s2vx_cnt=r.integers(0, 200, (batch, 20)).astype("f4"),
        strength=30.0,
        identity_context=r.random((batch, 10, DENSE_CTX)).astype("f4"),
        identity_context_raw=r.random((batch, 10, RAW)).astype("f4"),
        identity_context_support=r.integers(0, 500, (batch, 10)).astype("f4"),
    )


def _ctx(model: HGNNWinModel, inp: dict) -> torch.Tensor:
    dense = inp["identity_context"][..., 14:] if model.identity_conditioned_context.dense_dim else None
    with torch.no_grad():
        return model.identity_conditioned_context(
            inp["identity_context_raw"],
            inp["identity_context_support"],
            inp["champion_id"],
            inp["build_id"],
            dense,
        )


def _randomise(model: HGNNWinModel) -> None:
    with torch.no_grad():
        for p in model.identity_conditioned_context.context_projector.parameters():
            p.normal_()
        for p in model.identity_conditioned_context.identity_conditioner.parameters():
            p.normal_()


def test_enabled_only_with_flag_type_and_raw_dim() -> None:
    assert _model().identity_conditioned_context_enabled
    off = HGNNWinModel(
        HGNNConfig(
            n_champions=50,
            n_builds=5,
            identity_context_dim=DENSE_CTX,
            identity_context_raw_dim=RAW,
            use_identity_conditioned_context=True,
            identity_context_conditioning_type="none",
        )
    )
    assert not off.identity_conditioned_context_enabled
    # raw_dim == 0 also disables it even with the flag/type set.
    no_raw = HGNNWinModel(
        HGNNConfig(
            n_champions=50,
            n_builds=5,
            identity_context_dim=DENSE_CTX,
            identity_context_raw_dim=0,
            use_identity_conditioned_context=True,
            identity_context_conditioning_type="low_rank",
        )
    )
    assert not no_raw.identity_conditioned_context_enabled


def test_zero_init_is_opt_in() -> None:
    model = _model(source="raw_plus_dense")
    assert float(_ctx(model, _inputs()).abs().max()) == 0.0


def test_context_logit_is_antisymmetric_under_swap() -> None:
    for source in ("raw", "raw_plus_dense"):
        model = _model(source=source)
        _randomise(model)
        inp = _inputs(seed=1)
        sw = swap_hgnn_inputs(inp)
        fwd = _ctx(model, inp)
        rev = _ctx(model, sw)
        assert torch.allclose(rev, -fwd, atol=1e-5), source


def test_context_term_is_an_additive_residual_on_the_base_model() -> None:
    # The conditioned head is an additive residual: final_logit with the head
    # enabled minus the head's own context_logit must equal the head-disabled
    # ("none") base model, so the head changes nothing else. (The untrained base
    # is not itself antisymmetric, which is why we test the residual, not the
    # full logit, under swap.)
    model = _model(source="raw_plus_dense")
    _randomise(model)
    inp = _inputs(seed=2)
    base = HGNNWinModel(replace(model.config, use_identity_conditioned_context=False))
    base.load_state_dict(model.state_dict(), strict=False)
    base.eval()
    with torch.no_grad():
        full = model(**inp)["final_logit"]
        ctx = _ctx(model, inp)
        base_logit = base(**inp)["final_logit"]
    assert not base.identity_conditioned_context_enabled
    assert torch.allclose(full - ctx, base_logit, atol=1e-5)


def test_missing_identity_contributes_zero() -> None:
    model = _model()
    _randomise(model)
    zeros_raw = torch.zeros(4, 10, RAW)
    zeros_sup = torch.zeros(4, 10)
    cid = torch.zeros(4, 10, dtype=torch.long)
    bid = torch.zeros(4, 10, dtype=torch.long)
    with torch.no_grad():
        out = model.identity_conditioned_context(zeros_raw, zeros_sup, cid, bid)
    assert float(out.abs().max()) == 0.0


def test_zero_support_suppresses_and_low_support_is_dampened() -> None:
    model = _model()
    _randomise(model)
    inp = _inputs(seed=3)
    raw, cid, bid = inp["identity_context_raw"], inp["champion_id"], inp["build_id"]

    high = torch.full((8, 10), 1000.0)
    zero = torch.zeros(8, 10)
    low = torch.full((8, 10), 1.0)
    with torch.no_grad():
        c_high = model.identity_conditioned_context(raw, high, cid, bid)
        c_zero = model.identity_conditioned_context(raw, zero, cid, bid)
        c_low = model.identity_conditioned_context(raw, low, cid, bid)
    # Zero support across all players => exactly zero contribution.
    assert float(c_zero.abs().max()) == 0.0
    # Low support is strictly dampened relative to high support.
    assert float(c_low.abs().mean()) < float(c_high.abs().mean())


def test_support_gate_is_per_player() -> None:
    model = _model()
    _randomise(model)
    inp = _inputs(seed=4)
    raw, cid, bid = inp["identity_context_raw"], inp["champion_id"], inp["build_id"]
    sup = torch.full((8, 10), 500.0)

    def zero_one(player: int) -> torch.Tensor:
        s = sup.clone()
        s[:, player] = 0.0
        with torch.no_grad():
            return model.identity_conditioned_context(raw, s, cid, bid)

    with torch.no_grad():
        base = model.identity_conditioned_context(raw, sup, cid, bid)
    d0 = (zero_one(0) - base).abs()
    d1 = (zero_one(1) - base).abs()
    # Removing a single player's support changes the team aggregate (per-player
    # gating) and removing different players gives different deltas.
    assert float(d0.max()) > 0.0 and float(d1.max()) > 0.0
    assert not torch.allclose(d0, d1)


def test_lane_opponent_pairing_is_slot_aligned() -> None:
    # Permuting the enemy team's slot ORDER leaves the set summaries
    # (enemy_mean / enemy_weighted / ally_mean) unchanged but reassigns each
    # blue player's same-role lane opponent. If the head ignored slot pairing the
    # output would be unchanged; a change proves lane_opp is index-aligned.
    model = _model()
    _randomise(model)
    inp = _inputs(seed=5)
    raw = inp["identity_context_raw"].clone()
    cid = inp["champion_id"].clone()
    bid = inp["build_id"].clone()
    sup = torch.full((8, 10), 500.0)  # equal support -> gate is constant

    base = model.identity_conditioned_context(raw, sup, cid, bid)

    # Swap red slots 0 and 1 (indices 5 and 6) across every per-slot input.
    perm = list(range(10))
    perm[5], perm[6] = perm[6], perm[5]
    raw_p = raw[:, perm]
    cid_p = cid[:, perm]
    bid_p = bid[:, perm]
    permuted = model.identity_conditioned_context(raw_p, sup, cid_p, bid_p)
    assert not torch.allclose(base, permuted, atol=1e-6)


def test_residual_mlp_variant_runs_and_is_antisymmetric() -> None:
    model = _model(source="raw", residual_mlp=True)
    with torch.no_grad():
        for p in model.identity_conditioned_context.residual_mlp.parameters():
            p.normal_()
    _randomise(model)
    inp = _inputs(seed=6)
    fwd = _ctx(model, inp)
    rev = _ctx(model, swap_hgnn_inputs(inp))
    assert torch.allclose(rev, -fwd, atol=1e-5)
