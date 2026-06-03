"""End-to-end check that training gathers sidecar latents from the frozen
artifact (v28 cache) instead of per-game arrays."""

from __future__ import annotations

import json

import numpy as np

from app.core.utils.common import POSITIONS
from app.ml.cache_layout import ARRAY_FILES, ARRAY_SHAPES, CACHE_FORMAT, DISK_DTYPES
from app.ml.config import DatasetConfig, TrainConfig
from app.ml.encoder_sidecar import build_encoder_sidecar_metadata, save_encoder_sidecar
from app.ml.hgnn_model import load_hgnn_model
from app.ml.train import train

SIDE_DIMS = {"static": 2, "full_game": 3, "temporal": 4}


def _write_v28_cache(cache_dir, artifact_path, *, n_games=12, n_champions=12, rng):
    splits = {"train": 8, "val": 2, "test": 2}
    build_vocab = ["b0", "b1", "b2"]
    n_builds = len(build_vocab)

    champion_id = rng.integers(1, n_champions, size=(n_games, 10)).astype(np.int16)
    build_id = rng.integers(0, n_builds, size=(n_games, 10)).astype(np.int16)
    # Both classes in every split (val/test have exactly two rows).
    blue_win = np.tile([0, 1], n_games // 2).astype(np.uint8)

    arrays = {
        "win_rate": rng.uniform(0.4, 0.6, (n_games, 10)),
        "matchup_1v1": rng.uniform(0.4, 0.6, (n_games, 25)),
        "synergy_2vx": rng.uniform(0.4, 0.6, (n_games, 20)),
        "p1_cnt": rng.uniform(1, 80, (n_games, 10)),
        "m1v1_cnt": rng.uniform(1, 80, (n_games, 25)),
        "s2vx_cnt": rng.uniform(1, 80, (n_games, 20)),
        "m1v1_eff_n": rng.uniform(1, 80, (n_games, 25)),
        "s2vx_eff_n": rng.uniform(1, 80, (n_games, 20)),
        "champion_id": champion_id,
        "build_id": build_id,
        "blue_win": blue_win,
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, filename in ARRAY_FILES.items():
        assert arrays[name].shape[1:] == ARRAY_SHAPES[name]
        np.save(cache_dir / filename, arrays[name].astype(DISK_DTYPES[name], copy=False))

    # Frozen artifact covering every (champ, role, build) identity in the games.
    roles = list(POSITIONS) * 2
    identities = sorted(
        {
            (int(champion_id[g, s]), roles[s], build_vocab[int(build_id[g, s])])
            for g in range(n_games)
            for s in range(10)
        }
    )
    n_rows = len(identities)
    save_encoder_sidecar(
        artifact_path,
        champion_id=np.array([c for c, _, _ in identities], dtype=np.int32),
        teamposition=np.array([r for _, r, _ in identities]),
        build=np.array([b for _, _, b in identities]),
        static_latents=rng.normal(size=(n_rows, SIDE_DIMS["static"])).astype(np.float32),
        full_game_latents=rng.normal(size=(n_rows, SIDE_DIMS["full_game"])).astype(np.float32),
        temporal_latents=rng.normal(size=(n_rows, SIDE_DIMS["temporal"])).astype(np.float32),
        support=rng.uniform(0, 80, n_rows).astype(np.float32),
        metadata=build_encoder_sidecar_metadata(
            static_features=("base_health",),
            full_game_features=("damage_per_min",),
            temporal_features=("minute_0_damage",),
            split_metadata={"fit_split": "train"},
            encoder_configs={"static": {}, "full_game": {}, "temporal": {}},
            extra={"static_encoder": {"source": "deterministic champion dictionary"}},
        ),
    )

    offset = 0
    split_ranges = {}
    for name in ("train", "val", "test"):
        split_ranges[name] = {"start": offset, "stop": offset + splits[name]}
        offset += splits[name]
    (cache_dir / "cache_meta.json").write_text(
        json.dumps(
            {
                "format": CACHE_FORMAT,
                "n_games": n_games,
                "splits": splits,
                "split_order": ["train", "val", "test"],
                "split_ranges": split_ranges,
                "identity": {
                    "n_champions": n_champions,
                    "n_builds": n_builds,
                    "build_vocab": build_vocab,
                },
                "identity_encoder_sidecar": {
                    "path": str(artifact_path),
                    "dims": {**SIDE_DIMS, "total": sum(SIDE_DIMS.values())},
                    "metadata": {},
                },
            }
        )
    )


def test_train_gathers_sidecar_from_artifact_without_per_game_arrays(tmp_path) -> None:
    rng = np.random.default_rng(0)
    cache_dir = tmp_path / "cache"
    artifact_path = tmp_path / "sidecar.npz"
    _write_v28_cache(cache_dir, artifact_path, rng=rng)

    # No per-game sidecar arrays were written: the v28 gather path must supply them.
    assert not (cache_dir / "identity_full_game_sidecar.npy").exists()

    model_path = tmp_path / "model.pt"
    train(
        DatasetConfig(cache_dir=cache_dir, encoder_sidecar_path=artifact_path),
        TrainConfig(
            model_path=model_path,
            metrics_path=tmp_path / "metrics.json",
            max_epochs=1,
            patience=1,
            device="cpu",
        ),
        model_overrides={"use_identity_semantic_context_head": True},
    )

    assert model_path.exists()
    model, config, _ = load_hgnn_model(model_path)
    assert config.use_identity_semantic_context_head is True
    assert config.identity_full_game_sidecar_dim == SIDE_DIMS["full_game"]
    assert model.identity_semantic_context is not None
