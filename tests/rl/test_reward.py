from __future__ import annotations

import pytest

from app.rl.reward import RoleBuildConfig, resolve_rewards


BLUE = [1, 2, 3, 4, 5]
RED = [6, 7, 8, 9, 10]
POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


def _config(team: list[int], *, probability: float, build: int) -> RoleBuildConfig:
    return RoleBuildConfig(
        roles={champion: role for champion, role in zip(team, POSITIONS, strict=True)},
        builds={champion: build for champion in team},
        probability=probability,
    )


def test_expected_reward_uses_role_build_probabilities() -> None:
    blue_configs = [
        _config(BLUE, probability=0.9, build=0),
        _config(BLUE, probability=0.1, build=1),
    ]
    red_configs = [_config(RED, probability=1.0, build=0)]

    def sampler(team: list[int], side: str) -> list[RoleBuildConfig]:
        return blue_configs if side == "blue" else red_configs

    def predictor(*args) -> float:
        blue_builds = args[4]
        return 0.2 if set(blue_builds.values()) == {0} else 0.8

    result = resolve_rewards(BLUE, RED, predictor, sampler, "expected_value")

    assert result.p_blue_win_for_blue == pytest.approx(0.26)
    assert result.blue_reward == pytest.approx(-0.24)


def test_role_build_probabilities_are_validated() -> None:
    configs = [_config(BLUE, probability=0.0, build=0)]

    def sampler(_team: list[int], _side: str) -> list[RoleBuildConfig]:
        return configs

    with pytest.raises(ValueError, match="probabilities must sum"):
        resolve_rewards(BLUE, RED, lambda *args: 0.5, sampler, "expected_value")
