from __future__ import annotations

import logging
from typing import cast

import numpy as np

from app.ml.config import POSITIONS, UNK_INDEX

N_PLAYER_TOKENS = 10
PLAYER_TOKEN_LAYOUT = tuple(
    {"token_idx": i, "side": "blue", "role": role} for i, role in enumerate(POSITIONS)
) + tuple(
    {"token_idx": i + len(POSITIONS), "side": "red", "role": role}
    for i, role in enumerate(POSITIONS)
)
PLAYER_SPARSITY_KEYS = (
    "champion_unknown_by_token",
    "role_unknown_by_token",
    "build_unknown_by_token",
    "any_unknown_by_token",
)


def _empty_token_counts(
    token_count: int,
    keys: tuple[str, ...],
) -> dict[str, object]:
    return {
        "games": 0,
        **{key: np.zeros(token_count, dtype=np.int64) for key in keys},
    }


def _merge_token_counts(
    target: dict[str, object],
    source: dict[str, object],
    keys: tuple[str, ...],
) -> None:
    target["games"] = _count_int(target, "games") + _count_int(source, "games")
    for key in keys:
        target[key] = np.asarray(target[key], dtype=np.int64) + np.asarray(
            source[key], dtype=np.int64
        )


def _rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _count_int(counts: dict[str, object], key: str) -> int:
    return int(cast(int, counts[key]))


def _summary_float(summary: dict[str, object], key: str) -> float:
    return float(cast(float, summary[key]))


def empty_player_sparsity_counts() -> dict[str, object]:
    return _empty_token_counts(N_PLAYER_TOKENS, PLAYER_SPARSITY_KEYS)


def collect_player_sparsity_counts(
    champion_idx: np.ndarray,
    role_idx: np.ndarray,
    build_idx: np.ndarray,
) -> dict[str, object]:
    counts = empty_player_sparsity_counts()
    champion_unknown = champion_idx == UNK_INDEX
    role_unknown = role_idx == UNK_INDEX
    build_unknown = build_idx == UNK_INDEX
    any_unknown = champion_unknown | role_unknown | build_unknown

    counts["games"] = int(champion_idx.shape[0])
    counts["champion_unknown_by_token"] = champion_unknown.sum(axis=0).astype(np.int64)
    counts["role_unknown_by_token"] = role_unknown.sum(axis=0).astype(np.int64)
    counts["build_unknown_by_token"] = build_unknown.sum(axis=0).astype(np.int64)
    counts["any_unknown_by_token"] = any_unknown.sum(axis=0).astype(np.int64)
    return counts


def merge_player_sparsity_counts(
    target: dict[str, object],
    source: dict[str, object],
) -> None:
    _merge_token_counts(target, source, PLAYER_SPARSITY_KEYS)


def summarize_player_sparsity_counts(
    counts: dict[str, object],
) -> dict[str, object]:
    games = _count_int(counts, "games")
    slots = games * N_PLAYER_TOKENS
    champion_unknown = np.asarray(counts["champion_unknown_by_token"], dtype=np.int64)
    role_unknown = np.asarray(counts["role_unknown_by_token"], dtype=np.int64)
    build_unknown = np.asarray(counts["build_unknown_by_token"], dtype=np.int64)
    any_unknown = np.asarray(counts["any_unknown_by_token"], dtype=np.int64)

    champion_unknown_total = int(champion_unknown.sum())
    role_unknown_total = int(role_unknown.sum())
    build_unknown_total = int(build_unknown.sum())
    any_unknown_total = int(any_unknown.sum())

    return {
        "games": games,
        "slots": slots,
        "champion_unknown_slots": champion_unknown_total,
        "champion_unknown_frac": _rate(champion_unknown_total, slots),
        "role_unknown_slots": role_unknown_total,
        "role_unknown_frac": _rate(role_unknown_total, slots),
        "build_unknown_slots": build_unknown_total,
        "build_unknown_frac": _rate(build_unknown_total, slots),
        "any_unknown_slots": any_unknown_total,
        "any_unknown_frac": _rate(any_unknown_total, slots),
        "all_known_slots": slots - any_unknown_total,
        "all_known_frac": _rate(slots - any_unknown_total, slots),
        "by_token": [
            {
                **PLAYER_TOKEN_LAYOUT[i],
                "champion_unknown": int(champion_unknown[i]),
                "champion_unknown_frac": _rate(int(champion_unknown[i]), games),
                "role_unknown": int(role_unknown[i]),
                "role_unknown_frac": _rate(int(role_unknown[i]), games),
                "build_unknown": int(build_unknown[i]),
                "build_unknown_frac": _rate(int(build_unknown[i]), games),
                "any_unknown": int(any_unknown[i]),
                "any_unknown_frac": _rate(int(any_unknown[i]), games),
                "all_known": games - int(any_unknown[i]),
                "all_known_frac": _rate(games - int(any_unknown[i]), games),
            }
            for i in range(N_PLAYER_TOKENS)
        ],
    }


def log_player_sparsity_summary(
    logger: logging.Logger,
    split_name: str,
    counts: dict[str, object],
) -> None:
    summary = summarize_player_sparsity_counts(counts)
    logger.info(
        (
            "Player token sparsity %s: games=%d champion_unk=%d/%d %.2f%% "
            "role_unk=%d/%d %.2f%% build_unk=%d/%d %.2f%% any_unk=%d/%d %.2f%%"
        ),
        split_name,
        summary["games"],
        summary["champion_unknown_slots"],
        summary["slots"],
        _summary_float(summary, "champion_unknown_frac") * 100.0,
        summary["role_unknown_slots"],
        summary["slots"],
        _summary_float(summary, "role_unknown_frac") * 100.0,
        summary["build_unknown_slots"],
        summary["slots"],
        _summary_float(summary, "build_unknown_frac") * 100.0,
        summary["any_unknown_slots"],
        summary["slots"],
        _summary_float(summary, "any_unknown_frac") * 100.0,
    )


def cache_sparsity_metadata(
    overall_player_counts: dict[str, object],
    split_player_counts: dict[str, dict[str, object]],
) -> dict[str, object]:
    return {
        "player_token_sparsity": {
            "scope": "player_embedding_indices",
            "notes": [
                "champion_unknown, role_unknown, and build_unknown count "
                "player-token embedding ids equal to UNK_INDEX / 0.",
                "any_unknown counts player slots where at least one of "
                "champion_idx, role_idx, or build_idx is 0.",
            ],
            "overall": summarize_player_sparsity_counts(overall_player_counts),
            "splits": {
                split_name: summarize_player_sparsity_counts(
                    split_player_counts[split_name]
                )
                for split_name in ("train", "val", "test")
            },
        },
    }
