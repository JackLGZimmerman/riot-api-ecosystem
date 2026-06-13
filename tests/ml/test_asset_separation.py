from __future__ import annotations

import numpy as np
import pytest

from app.ml.build_catalog import (
    ASSET_FAMILY_PREGAME_NATIVE,
    KEY_SPACE_CHAMP_ROLE,
    BuildCatalog,
    assert_pregame_native_key_space,
    build_catalog,
)

VOCAB = ("a", "b", "c")


def _p1() -> dict[tuple[int, str, str], tuple[float, int]]:
    return {
        (1, "TOP", "a"): (0.52, 80),
        (1, "TOP", "b"): (0.48, 20),
        (2, "TOP", "a"): (0.51, 900),
        (2, "TOP", "b"): (0.49, 100),
    }


def test_real_catalog_is_pregame_native() -> None:
    catalog = build_catalog(_p1(), VOCAB)
    catalog.assert_pregame_native()  # must not raise
    assert BuildCatalog.asset_family == ASSET_FAMILY_PREGAME_NATIVE
    assert BuildCatalog.key_space == KEY_SPACE_CHAMP_ROLE


def test_key_space_guard_accepts_champ_role() -> None:
    assert_pregame_native_key_space([(1, "TOP"), (2, "JUNGLE")])  # must not raise
    assert_pregame_native_key_space([(np.int64(3), "MID")])  # numpy int accepted


def test_key_space_guard_rejects_build_key() -> None:
    with pytest.raises(ValueError, match="pregame-native"):
        assert_pregame_native_key_space([(1, "TOP", "a")])


def test_key_space_guard_rejects_bad_types() -> None:
    with pytest.raises(ValueError, match="pregame-native"):
        assert_pregame_native_key_space([("x", "TOP")])
