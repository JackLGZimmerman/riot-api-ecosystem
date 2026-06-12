from __future__ import annotations

import pytest

from app.rl.pool import ChampionPool, PoolEntry
from app.rl.reward import RoleBuildConfig, make_pool_sampler, resolve_rewards


BLUE = [1, 2, 3, 4, 5]
RED = [6, 7, 8, 9, 10]
POSITIONS = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")


def _config(
    team: list[int],
    *,
    probability: float,
    build: int,
    retained_mass: float = 1.0,
) -> RoleBuildConfig:
    return RoleBuildConfig(
        roles={champion: role for champion, role in zip(team, POSITIONS, strict=True)},
        builds={champion: build for champion in team},
        probability=probability,
        retained_mass=retained_mass,
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


def test_result_surfaces_min_retained_mass_per_side() -> None:
    blue_configs = [
        _config(BLUE, probability=0.9, build=0, retained_mass=0.8),
        _config(BLUE, probability=0.1, build=1, retained_mass=0.8),
    ]
    red_configs = [_config(RED, probability=1.0, build=0, retained_mass=0.95)]

    def sampler(team: list[int], side: str) -> list[RoleBuildConfig]:
        return blue_configs if side == "blue" else red_configs

    result = resolve_rewards(BLUE, RED, lambda *args: 0.5, sampler, "expected_value")

    assert result.blue_retained_mass == pytest.approx(0.8)
    assert result.red_retained_mass == pytest.approx(0.95)


def test_worst_case_min_probability_masks_implausible_tail() -> None:
    blue_configs = [
        _config(BLUE, probability=0.9, build=0),
        _config(BLUE, probability=0.1, build=1),
    ]
    red_configs = [_config(RED, probability=1.0, build=0)]

    def sampler(team: list[int], side: str) -> list[RoleBuildConfig]:
        return blue_configs if side == "blue" else red_configs

    def predictor(*args) -> float:
        blue_builds = args[4]
        return 0.1 if set(blue_builds.values()) == {1} else 0.6

    # Default keeps every config: the implausible tail dominates the min.
    default = resolve_rewards(BLUE, RED, predictor, sampler, "worst_case")
    assert default.p_blue_win_for_blue == pytest.approx(0.1)

    masked = resolve_rewards(
        BLUE, RED, predictor, sampler, "worst_case", worst_case_min_probability=0.5
    )
    assert masked.p_blue_win_for_blue == pytest.approx(0.6)

    # A threshold that would empty a side falls back to the full set.
    fallback = resolve_rewards(
        BLUE, RED, predictor, sampler, "worst_case", worst_case_min_probability=1.5
    )
    assert fallback.p_blue_win_for_blue == pytest.approx(0.1)


def test_pool_sampler_stamps_retained_mass() -> None:
    entries = {
        champion: (
            PoolEntry(role=role, build_id=0, weight=0.9),
            PoolEntry(role=role, build_id=1, weight=0.1),
        )
        for champion, role in zip(BLUE, POSITIONS, strict=True)
    }
    sampler = make_pool_sampler(ChampionPool(entries=entries), 1)

    configs = sampler(BLUE, "blue")

    assert len(configs) == 1
    assert configs[0].probability == pytest.approx(1.0)
    # Top-1 mass over the full enumerated mass: 0.9^5 / 1.0
    assert configs[0].retained_mass == pytest.approx(0.9**5)
