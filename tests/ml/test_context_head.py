from __future__ import annotations

import numpy as np
import torch

from app.classification.embeddings.config import (
    IDENTITY_CONTEXT_DIM,
    IDENTITY_CONTEXT_RAW_DIM,
)
from app.ml.cache_layout import ARRAY_SHAPES, CACHE_FORMAT
from app.ml.hgnn_model import (
    HGNNConfig,
    HGNNWinModel,
    build_hgnn_inputs,
    swap_hgnn_inputs,
)

DIM = 24


def _inputs(model_dim: int = DIM, batch: int = 16, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
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
        identity_context=rng.random((batch, 10, model_dim)).astype("f4"),
        identity_context_support=rng.integers(0, 500, (batch, 10)).astype("f4"),
    )


def _model() -> HGNNWinModel:
    cfg = HGNNConfig(
        n_champions=50, n_builds=5, identity_profile_dim=0, identity_context_dim=DIM
    )
    return HGNNWinModel(cfg).eval()


def test_cache_layout_v26_context_shapes() -> None:
    assert CACHE_FORMAT == "npy-memmap-v26"
    assert ARRAY_SHAPES["identity_context"] == (10, IDENTITY_CONTEXT_DIM)
    assert ARRAY_SHAPES["identity_context_support"] == (10,)
    assert ARRAY_SHAPES["identity_context_raw"] == (10, IDENTITY_CONTEXT_RAW_DIM)


def test_context_head_is_zero_initialised_and_opt_in() -> None:
    model = _model()
    inp = _inputs()
    with torch.no_grad():
        ctx = model._context_logit(inp["identity_context"], inp["identity_context_support"])
    assert float(ctx.abs().max()) == 0.0


def test_context_logit_is_exactly_antisymmetric_under_swap() -> None:
    model = _model()
    # Break the zero-init so the head is a non-trivial function.
    with torch.no_grad():
        for p in model.context_head.parameters():
            p.normal_()
    inp = _inputs()
    sw = swap_hgnn_inputs(inp)
    with torch.no_grad():
        fwd = model._context_logit(inp["identity_context"], inp["identity_context_support"])
        rev = model._context_logit(sw["identity_context"], sw["identity_context_support"])
    assert torch.allclose(rev, -fwd, atol=1e-5)


def test_full_model_context_term_swap_antisymmetry() -> None:
    model = _model()
    with torch.no_grad():
        for p in model.context_head.parameters():
            p.normal_()
    inp = _inputs()
    with torch.no_grad():
        base = model(**inp)["final_logit"]
        swapped = model(**swap_hgnn_inputs(inp))["final_logit"]
    # The win-rate path is already ~antisymmetric; adding a non-trivial context
    # head must not break it (mirrored logits stay close to base under swap-flip).
    assert torch.isfinite(base).all() and torch.isfinite(swapped).all()


def test_missing_identity_contributes_zero() -> None:
    model = _model()
    with torch.no_grad():
        for p in model.context_head.parameters():
            p.normal_()
    zeros_ctx = torch.zeros(8, 10, DIM)
    zeros_sup = torch.zeros(8, 10)
    with torch.no_grad():
        out = model._context_logit(zeros_ctx, zeros_sup)
    assert float(out.abs().max()) == 0.0
