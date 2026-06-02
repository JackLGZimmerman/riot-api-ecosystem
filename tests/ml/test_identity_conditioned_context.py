"""Unit tests for the low-rank identity-conditioned context head.

Covers the acceptance-critical invariants: zero-init opt-in, blue/red
antisymmetry under team swap, per-player support gating, slot/lane-opponent
alignment, and missing-identity behaviour.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from app.ml.hgnn_model import (
    HGNNConfig,
    HGNNWinModel,
    build_hgnn_inputs,
    swap_hgnn_inputs,
)

RAW = 62
DENSE_CTX = 24  # 14 interpretable + 10 dense tail


def _model(
    *,
    source: str = "raw",
    residual_mlp: bool = False,
    conditioning_type: str = "low_rank",
    context_set_encoder_type: str = "mean",
    include_products: bool = False,
    include_support_features: bool = False,
) -> HGNNWinModel:
    cfg = HGNNConfig(
        n_champions=50,
        n_builds=5,
        identity_profile_dim=0,
        identity_context_dim=DENSE_CTX,
        context_interpretable_dim=14,
        identity_context_raw_dim=RAW,
        use_identity_conditioned_context=True,
        identity_context_conditioning_type=conditioning_type,
        identity_context_source=source,
        identity_context_rank=8,
        identity_context_hidden_dim=32,
        identity_context_emb_dim=16,
        identity_context_use_residual_mlp=residual_mlp,
        identity_context_include_products=include_products,
        identity_context_include_support_features=include_support_features,
        context_set_encoder_type=context_set_encoder_type,
    )
    return HGNNWinModel(cfg).eval()


def _inputs(batch: int = 8, seed: int = 0, *, include_relationship_features: bool = False) -> dict:
    r = np.random.default_rng(seed)
    return build_hgnn_inputs(
        champion_id=r.integers(0, 50, (batch, 10)),
        build_id=r.integers(0, 5, (batch, 10)),
        win_rate=r.random((batch, 10)).astype("f4"),
        p1_cnt=r.integers(0, 200, (batch, 10)).astype("f4"),
        strength=30.0,
        matchup_1v1=r.random((batch, 25)).astype("f4"),
        synergy_2vx=r.random((batch, 20)).astype("f4"),
        m1v1_cnt=r.integers(0, 200, (batch, 25)).astype("f4"),
        s2vx_cnt=r.integers(0, 200, (batch, 20)).astype("f4"),
        include_relationship_features=include_relationship_features,
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
        ctx = model.identity_conditioned_context
        modules = (
            (ctx.context_projector, ctx.identity_conditioner)
            if ctx.conditioning_type == "low_rank"
            else (ctx.film_conditioner, ctx.film_scorer)
        )
        for module in modules:
            for p in module.parameters():
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


def test_forward_exposes_context_logit_decomposition() -> None:
    model = _model()
    inp = _inputs(batch=4)
    _randomise(model)

    with torch.no_grad():
        out = model(**inp)

    assert {"final_logit", "base_logit", "context_logit"} <= set(out)
    assert torch.allclose(out["final_logit"], out["base_logit"] + out["context_logit"])
    assert torch.allclose(out["context_logit"], _ctx(model, inp))


def test_hgnn_model_has_no_direct_prior_shortcut_capacity() -> None:
    model = HGNNWinModel(
        HGNNConfig(
            n_champions=50,
            n_builds=5,
            identity_profile_dim=0,
            identity_context_dim=0,
            identity_context_raw_dim=0,
        )
    )

    assert not hasattr(model, "prior_shortcut")
    assert not hasattr(model, "prior_shortcut_residual")


def test_team_slot_readout_is_zero_init_opt_in_capacity() -> None:
    model = HGNNWinModel(
        HGNNConfig(
            n_champions=50,
            n_builds=5,
            identity_profile_dim=0,
            identity_context_dim=0,
            identity_context_raw_dim=0,
            team_slot_readout_hidden=(16,),
        )
    )

    assert model.team_slot_readout is not None
    final_layer = model.team_slot_readout[-1]
    assert torch.allclose(final_layer.weight, torch.zeros_like(final_layer.weight))
    assert torch.allclose(final_layer.bias, torch.zeros_like(final_layer.bias))

    team = torch.randn(3, 5, model.config.node_dim)
    residual = model.team_slot_readout(team.flatten(start_dim=1))
    assert torch.allclose(residual, torch.zeros_like(residual))


def test_team_readout_uses_mean_and_attention_without_max_pool() -> None:
    model = HGNNWinModel(
        HGNNConfig(
            n_champions=50,
            n_builds=5,
            identity_profile_dim=0,
            identity_context_dim=0,
            identity_context_raw_dim=0,
        )
    ).eval()
    team = torch.randn(3, 5, model.config.node_dim)

    assert model.team_proj.in_features == model.config.node_dim * 2
    with torch.no_grad():
        expected = model.team_proj(
            torch.cat([team.mean(dim=1), model.attn_pool(team)], dim=-1)
        )
        actual = model._readout(team)

    assert torch.allclose(actual, expected)


def test_base_logit_is_built_from_blue_and_red_team_vectors() -> None:
    model = HGNNWinModel(
        HGNNConfig(
            n_champions=50,
            n_builds=5,
            identity_profile_dim=0,
            identity_context_dim=0,
            identity_context_raw_dim=0,
        )
    ).eval()
    inp = _inputs(batch=4, seed=11)
    a = torch.randn(4, model.config.node_dim)
    b = torch.randn(4, model.config.node_dim)
    calls: list[torch.Size] = []

    def fake_readout(team: torch.Tensor) -> torch.Tensor:
        calls.append(team.shape)
        return a if len(calls) == 1 else b

    model._readout = fake_readout  # type: ignore[method-assign]

    with torch.no_grad():
        out = model(**inp)
        relationship_placeholder = torch.zeros_like(a)
        expected_base = model.head(
            torch.cat([a, b, a - b, a * b, relationship_placeholder], dim=-1)
        ).squeeze(-1)

    assert calls == [torch.Size([4, 5, model.config.node_dim])] * 2
    assert torch.allclose(out["base_logit"], expected_base)
    assert torch.allclose(out["final_logit"], out["base_logit"] + out["context_logit"])


def test_default_relationship_removed_path_ignores_1v1_and_2vx_inputs() -> None:
    model = HGNNWinModel(
        HGNNConfig(
            n_champions=50,
            n_builds=5,
            identity_profile_dim=0,
            identity_context_dim=0,
            identity_context_raw_dim=0,
        )
    ).eval()
    inp = _inputs(batch=4, seed=3)
    assert "delta_logit_2vx" not in inp
    assert "delta_logit_1v1" not in inp
    assert "conf_2vx" not in inp
    assert "conf_1v1" not in inp

    changed = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in inp.items()
    }
    changed["mu_2vx"] = torch.full_like(changed["mu_2vx"], 0.01)
    changed["mu_1v1"] = torch.full_like(changed["mu_1v1"], 0.99)

    with torch.no_grad():
        base = model(**inp)
        altered = model(**changed)

    for key in ("final_logit", "base_logit", "context_logit"):
        assert torch.allclose(altered[key], base[key])


def test_relationship_features_are_explicit_opt_in_inputs() -> None:
    inp = _inputs(batch=3, seed=4, include_relationship_features=True)

    assert {"delta_logit_2vx", "delta_logit_1v1", "conf_2vx", "conf_1v1"} <= set(inp)


def test_1vx_support_omits_redundant_missing_flag() -> None:
    inp = build_hgnn_inputs(
        champion_id=np.zeros((1, 10), dtype="i8"),
        build_id=np.zeros((1, 10), dtype="i8"),
        win_rate=np.full((1, 10), 0.5, dtype="f4"),
        p1_cnt=np.array([[0.0, 10.0] + [1.0] * 8], dtype="f4"),
        strength=30.0,
    )

    assert "missing_1vx" not in inp
    assert inp["conf_1vx"][0, 0].item() == 0.0
    assert inp["log_count_1vx"][0, 0].item() == 0.0
    assert inp["conf_1vx"][0, 1].item() > 0.0
    assert inp["log_count_1vx"][0, 1].item() > 0.0

    model = HGNNWinModel(
        HGNNConfig(
            n_champions=50,
            n_builds=5,
            identity_profile_dim=0,
            identity_context_dim=0,
            identity_context_raw_dim=0,
        )
    )
    assert model.phi["1vx"].value[0].in_features == 4
    assert model.phi["1vx"].gate[0].in_features == 3


def test_1vx_variance_ablation_removes_variance_encoder_inputs() -> None:
    model = HGNNWinModel(
        HGNNConfig(
            n_champions=50,
            n_builds=5,
            identity_profile_dim=0,
            identity_context_dim=0,
            identity_context_raw_dim=0,
            use_1vx_posterior_variance=False,
        )
    )
    inp = _inputs(batch=2, seed=12)

    assert model.phi["1vx"].value[0].in_features == 3
    assert model.phi["1vx"].gate[0].in_features == 2

    with torch.no_grad():
        out = model(**inp)

    assert out["final_logit"].shape == (2,)


def test_build_inputs_accepts_loader_without_relationship_tables() -> None:
    rng = np.random.default_rng(4)
    inp = build_hgnn_inputs(
        champion_id=rng.integers(0, 50, (2, 10)),
        build_id=rng.integers(0, 5, (2, 10)),
        win_rate=rng.random((2, 10)).astype("f4"),
        p1_cnt=rng.integers(0, 200, (2, 10)).astype("f4"),
        strength=30.0,
    )

    assert inp["mu_1v1"].shape == (2, 25)
    assert inp["mu_2vx"].shape == (2, 20)
    assert torch.allclose(inp["mu_1v1"], torch.full_like(inp["mu_1v1"], 0.5))
    assert torch.allclose(inp["mu_2vx"], torch.full_like(inp["mu_2vx"], 0.5))
    assert "delta_logit_1v1" not in inp
    assert "delta_logit_2vx" not in inp


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


def test_product_features_are_global_opt_in_interactions() -> None:
    base = _model()
    model = _model(include_products=True)
    ctx = model.identity_conditioned_context
    assert ctx.context_feature_dim == base.identity_conditioned_context.context_feature_dim + 7

    self_src = torch.zeros(2, 5, RAW)
    enemy_src = torch.zeros(2, 5, RAW)
    self_src[..., 3] = 2.0   # armor
    self_src[..., 4] = 3.0   # magic resist
    self_src[..., 5] = 5.0   # damage pressure
    self_src[..., 9] = 4.0   # damage taken
    self_src[..., 10] = 6.0  # heal/shield
    enemy_src[..., 0] = 0.7
    enemy_src[..., 1] = 0.2
    enemy_src[..., 5] = 0.5
    enemy_src[..., 10] = 0.3

    with torch.no_grad():
        products = ctx._context_features(self_src, enemy_src)[..., -7:]
    expected = torch.tensor([1.4, 0.6, 1.0, -1.5, 2.0, 1.5, 30.0])
    assert torch.allclose(products, expected.view(1, 1, 7).expand_as(products))


def test_product_variant_remains_antisymmetric_under_swap() -> None:
    model = _model(include_products=True)
    _randomise(model)
    inp = _inputs(seed=9)
    fwd = _ctx(model, inp)
    rev = _ctx(model, swap_hgnn_inputs(inp))
    assert torch.allclose(rev, -fwd, atol=1e-5)


def test_support_features_are_global_opt_in_conditioning() -> None:
    base = _model()
    model = _model(include_support_features=True)
    base_ctx = base.identity_conditioned_context
    ctx = model.identity_conditioned_context

    assert ctx.context_feature_dim == base_ctx.context_feature_dim + 15
    assert ctx.identity_conditioner[0].in_features == (
        base_ctx.identity_conditioner[0].in_features + 3
    )

    raw = torch.zeros(1, 10, RAW)
    support = torch.tensor([[0.0, 30.0, 1000.0] + [5.0] * 7])
    with torch.no_grad():
        src = ctx._source(raw, None, support)
    log_scale = torch.log1p(torch.tensor(1000.0))
    expected = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.5, float(torch.log1p(torch.tensor(30.0)) / log_scale), 0.0],
            [
                1000.0 / 1030.0,
                1.0,
                0.0,
            ],
        ]
    )
    assert torch.allclose(src[0, :3, -3:], expected, atol=1e-6)


def test_support_feature_variant_remains_antisymmetric_under_swap() -> None:
    model = _model(include_support_features=True)
    _randomise(model)
    inp = _inputs(seed=10)
    fwd = _ctx(model, inp)
    rev = _ctx(model, swap_hgnn_inputs(inp))
    assert torch.allclose(rev, -fwd, atol=1e-5)


def test_product_variant_fails_early_without_raw_interpretable_prefix() -> None:
    with pytest.raises(ValueError, match="identity_context_include_products"):
        HGNNWinModel(
            HGNNConfig(
                n_champions=50,
                n_builds=5,
                identity_context_dim=5,
                identity_context_raw_dim=5,
                use_identity_conditioned_context=True,
                identity_context_conditioning_type="low_rank",
                identity_context_include_products=True,
            )
        )


def test_product_variant_fails_early_when_conditioned_head_is_disabled() -> None:
    with pytest.raises(ValueError, match="identity_context_include_products"):
        HGNNWinModel(
            HGNNConfig(
                n_champions=50,
                n_builds=5,
                identity_context_dim=DENSE_CTX,
                identity_context_raw_dim=RAW,
                use_identity_conditioned_context=False,
                identity_context_conditioning_type="none",
                identity_context_include_products=True,
            )
        )


def test_support_feature_variant_fails_early_when_conditioned_head_is_disabled() -> None:
    with pytest.raises(ValueError, match="identity_context_include_support_features"):
        HGNNWinModel(
            HGNNConfig(
                n_champions=50,
                n_builds=5,
                identity_context_dim=DENSE_CTX,
                identity_context_raw_dim=RAW,
                use_identity_conditioned_context=False,
                identity_context_conditioning_type="none",
                identity_context_include_support_features=True,
            )
        )


def test_support_feature_variant_fails_early_with_invalid_log_scale() -> None:
    with pytest.raises(ValueError, match="identity_context_support_log_scale"):
        HGNNWinModel(
            HGNNConfig(
                n_champions=50,
                n_builds=5,
                identity_context_dim=DENSE_CTX,
                identity_context_raw_dim=RAW,
                use_identity_conditioned_context=True,
                identity_context_conditioning_type="low_rank",
                identity_context_include_support_features=True,
                identity_context_support_log_scale=0.0,
            )
        )


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


def test_film_zero_init_and_regularization_start_at_zero() -> None:
    model = _model(conditioning_type="film")
    assert float(_ctx(model, _inputs()).abs().max()) == 0.0
    assert float(model.context_regularization_loss().detach()) == 0.0


def test_film_missing_or_zero_support_contributes_zero_and_low_support_dampens() -> None:
    model = _model(conditioning_type="film")
    _randomise(model)
    inp = _inputs(seed=7)
    raw, cid, bid = inp["identity_context_raw"], inp["champion_id"], inp["build_id"]
    high = torch.full((8, 10), 1000.0)
    low = torch.full((8, 10), 1.0)
    zero = torch.zeros(8, 10)
    with torch.no_grad():
        c_missing = model.identity_conditioned_context(raw, None, cid, bid)
        c_zero = model.identity_conditioned_context(raw, zero, cid, bid)
        c_low = model.identity_conditioned_context(raw, low, cid, bid)
        c_high = model.identity_conditioned_context(raw, high, cid, bid)
    assert float(c_missing.abs().max()) == 0.0
    assert float(c_zero.abs().max()) == 0.0
    assert float(c_low.abs().mean()) < float(c_high.abs().mean())


def test_film_context_logit_is_antisymmetric_under_swap() -> None:
    model = _model(conditioning_type="film")
    _randomise(model)
    inp = _inputs(seed=8)
    fwd = _ctx(model, inp)
    rev = _ctx(model, swap_hgnn_inputs(inp))
    assert torch.allclose(rev, -fwd, atol=1e-5)


def test_film_regularization_tracks_direct_modulation_parameters() -> None:
    model = _model(conditioning_type="film")
    assert float(model.context_regularization_loss().detach()) == 0.0
    with torch.no_grad():
        model.identity_conditioned_context.film_conditioner[-1].weight.fill_(0.5)
    assert float(model.context_regularization_loss().detach()) > 0.0
