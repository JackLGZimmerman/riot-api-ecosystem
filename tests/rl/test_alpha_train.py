from __future__ import annotations

import pytest

from app.rl.alpha_train import AlphaTrainConfig, _entrypoint_config, _serving_patch


def test_serving_patch_config_is_optional_for_non_patch_artifacts() -> None:
    assert _serving_patch(AlphaTrainConfig(top_k_build_configs=8)) is None


def test_serving_patch_config_requires_complete_pair() -> None:
    with pytest.raises(ValueError, match="must be set together"):
        _serving_patch(
            AlphaTrainConfig(top_k_build_configs=8, serving_patch_season=16)
        )
    with pytest.raises(ValueError, match="must be set together"):
        _serving_patch(AlphaTrainConfig(top_k_build_configs=8, serving_patch=11))


def test_serving_patch_config_returns_pair() -> None:
    cfg = AlphaTrainConfig(
        top_k_build_configs=8,
        serving_patch_season=16,
        serving_patch=11,
    )

    assert _serving_patch(cfg) == (16, 11)


def test_entrypoint_reads_serving_patch_environment(monkeypatch) -> None:
    monkeypatch.setenv("HGNN_SERVING_PATCH_SEASON", "16")
    monkeypatch.setenv("HGNN_SERVING_PATCH", "11")

    assert _serving_patch(_entrypoint_config()) == (16, 11)
