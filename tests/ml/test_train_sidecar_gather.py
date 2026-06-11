"""End-to-end checks for compact sidecar artifact gathering."""

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


def _write_compact_cache(cache_dir, artifact_path, *, n_games=12, n_champions=12, rng):
    splits = {"train": 8, "test": 4}
    build_vocab = ["b0", "b1", "b2"]
    n_builds = len(build_vocab)

    champion_id = rng.integers(1, n_champions, size=(n_games, 10)).astype(np.int16)
    build_id = rng.integers(0, n_builds, size=(n_games, 10)).astype(np.int16)
    # Alternating labels keep both classes in each split range.
    blue_win = np.tile([0, 1], n_games // 2).astype(np.uint8)

    arrays = {
        "win_rate": rng.uniform(0.4, 0.6, (n_games, 10)),
        "p1_cnt": rng.uniform(1, 80, (n_games, 10)),
        "champion_id": champion_id,
        "build_id": build_id,
        "blue_win": blue_win,
        "loadout_features": np.zeros((n_games, 10), dtype=np.float32),
        "patch_features": np.zeros((n_games, 2), dtype=np.float32),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, filename in ARRAY_FILES.items():
        assert arrays[name].shape[1:] == ARRAY_SHAPES[name]
        np.save(cache_dir / filename, arrays[name].astype(DISK_DTYPES[name], copy=False))
    np.save(
        cache_dir / "identity_context_raw.npy",
        rng.uniform(0.0, 1.0, (n_games, 10, 14)).astype(np.float32),
    )

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
    for name in ("train", "test"):
        split_ranges[name] = {"start": offset, "stop": offset + splits[name]}
        offset += splits[name]
    (cache_dir / "cache_meta.json").write_text(
        json.dumps(
            {
                "format": CACHE_FORMAT,
                "n_games": n_games,
                "splits": splits,
                "split_order": ["train", "test"],
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


def test_train_gathers_sidecar_for_learned_moe_from_artifact_without_per_game_arrays(
    tmp_path,
) -> None:
    rng = np.random.default_rng(0)
    cache_dir = tmp_path / "cache"
    artifact_path = tmp_path / "sidecar.npz"
    _write_compact_cache(cache_dir, artifact_path, rng=rng)

    # No per-game sidecar arrays were written: compact artifact gather supplies them.
    assert not (cache_dir / "identity_full_game_sidecar.npy").exists()

    model_path = tmp_path / "model.pt"
    metrics_path = tmp_path / "metrics.json"
    train(
        DatasetConfig(cache_dir=cache_dir, encoder_sidecar_path=artifact_path),
        TrainConfig(
            model_path=model_path,
            metrics_path=metrics_path,
            max_epochs=1,
            patience=1,
            device="cpu",
        ),
        model_overrides={
            "use_learned_semantic_moe": True,
            "node_dim": 16,
            "edge_hidden": 8,
            "value_hidden": (),
            "gate_hidden": (),
            "node_init_hidden": (),
            "readout_hidden": (),
            "semantic_moe_num_experts": 3,
            "semantic_moe_top_k": 2,
            "semantic_moe_factor_dim": 8,
            "semantic_moe_factor_hidden": (),
            "semantic_moe_router_hidden": (),
            "semantic_moe_expert_hidden": (),
            "semantic_moe_context_token_dropout": 0.0,
        },
    )

    assert model_path.exists()
    model, config, _ = load_hgnn_model(model_path)
    assert config.use_learned_semantic_moe is True
    assert config.identity_full_game_sidecar_dim == SIDE_DIMS["full_game"]
    assert model.learned_semantic_moe is not None

    metrics = json.loads(metrics_path.read_text())
    assert metrics["evaluated_splits"] == ["train", "test"]
    assert metrics["selection_split"] == "test"
    assert "val" not in metrics
    assert "semantic_moe_view_top_k" not in metrics["model_config"]
    assert "decision_threshold" not in metrics
    assert "temperature_scaling" not in metrics
    assert "best_checkpoint_test_ece" not in metrics
    for split_name in ("train", "test"):
        assert set(metrics[split_name]) == {"n", "accuracy", "nll"}
        assert "support_buckets" not in metrics[split_name]
        assert "temperature_scaled" not in metrics[split_name]
        assert "logit_diagnostics" not in metrics[split_name]
        assert "semantic_moe_diagnostics" not in metrics[split_name]


def test_train_gathers_sidecar_and_semantic_group_features_from_compact_cache(
    tmp_path,
) -> None:
    rng = np.random.default_rng(2)
    cache_dir = tmp_path / "cache"
    artifact_path = tmp_path / "sidecar.npz"
    _write_compact_cache(cache_dir, artifact_path, rng=rng)

    assert not (cache_dir / "semantic_group_features.npy").exists()

    model_path = tmp_path / "model.pt"
    metrics_path = tmp_path / "metrics.json"
    train(
        DatasetConfig(cache_dir=cache_dir, encoder_sidecar_path=artifact_path),
        TrainConfig(
            model_path=model_path,
            metrics_path=metrics_path,
            max_epochs=1,
            patience=1,
            device="cpu",
        ),
        model_overrides={
            "use_learned_semantic_moe": True,
            "use_semantic_group_features": True,
            "node_dim": 16,
            "edge_hidden": 8,
            "value_hidden": (),
            "gate_hidden": (),
            "node_init_hidden": (),
            "readout_hidden": (),
            "semantic_moe_num_experts": 3,
            "semantic_moe_top_k": 2,
            "semantic_moe_factor_dim": 8,
            "semantic_moe_factor_hidden": (),
            "semantic_moe_router_hidden": (),
            "semantic_moe_expert_hidden": (),
            "semantic_moe_context_token_dropout": 0.0,
        },
    )

    assert (cache_dir / "semantic_group_features.npy").exists()
    model, config, _ = load_hgnn_model(model_path)
    assert config.use_learned_semantic_moe is True
    assert config.use_semantic_group_features is True
    assert model.learned_semantic_moe is not None

    metrics = json.loads(metrics_path.read_text())
    for split_name in ("train", "test"):
        assert set(metrics[split_name]) == {"n", "accuracy", "nll"}
