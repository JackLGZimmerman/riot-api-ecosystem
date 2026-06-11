"""Regenerate the HGNN context examples audit with model prediction gaps."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch

from app.ml.audit_format import (
    _format_pct,
    _format_pp,
    _format_pp_mse,
    _max_or_nan,
    _mean_or_nan,
)
from app.ml.config import DatasetConfig
from app.ml.context_audit_lens import AuditLens
from app.ml.context_audit_specs import AuditSpec, BinSpec, audit_specs
from app.ml.dataset import SPLIT_ORDER, SplitData, identity_meta, load_splits
from app.ml.hgnn_model import (
    HGNNWinModel,
    build_hgnn_inputs,
    load_hgnn_model,
    resolve_device,
)
from app.ml.semantic_group_features import (
    BURST_DAMAGE_THRESHOLD,
    FOCUS_HP_LOW_THRESHOLD,
    HARD_CC_THRESHOLD,
    HEAVY_TAKEN_THRESHOLD,
    HIGH_HP_THRESHOLD,
    RANGED_ATTACK_RANGE_THRESHOLD,
    SEMANTIC_GROUP_FEATURE_DIM,
    SEMANTIC_GROUP_FEATURE_SCHEMA_VERSION,
    static_hp_range_lookups,
)
from app.ml.train import _SidecarGatherer, _build_sidecar_gatherer, _model_uses_sidecar

DEFAULT_CONTEXT_CACHE_DIR = Path("app/ml/data/cache")
DEFAULT_MODEL_CACHE_DIR = Path("app/ml/data/experiments/semantic_context_compact_cache")
DEFAULT_MODEL_PATH = Path(
    "app/ml/data/experiments/semantic_context_compact_run/model.pt"
)
DEFAULT_OUTPUT_PATH = Path("app/ml/documentation/HGNN_CONTEXT_EXAMPLES_AUDIT.md")
DEFAULT_PREDICTION_CACHE = Path(
    "app/ml/data/experiments/semantic_context_compact_run/audit_final_blue_probability.npy"
)

__all__ = [
    "AuditBin",
    "AuditData",
    "AuditRow",
    "AuditSpec",
    "AuditSplitSummary",
    "BinSpec",
    "FLAGGED_AUDIT_TITLES",
    "audit_json_payload",
    "audit_specs",
    "evaluate_specs",
    "evaluate_specs_with_bootstrap",
    "gap_summary",
    "write_audit_json",
]

AUDIT_SPLITS = ("all", *SPLIT_ORDER)
FLAGGED_AUDIT_TITLES = frozenset(
    {
        "Ezreal BOTTOM `attack_damage` vs enemy hard CC",
        "LeeSin JUNGLE `ad_off_tank` vs enemy magic",
        "Ambessa TOP `attack_damage` vs enemy damage",
        "MasterYi JUNGLE any build vs enemy hard CC",
        "Vayne BOTTOM `on_hit` vs enemy frontline count",
        "Karma UTILITY any build vs enemy frontline count",
    }
)


def _audit_split_range(meta: dict, audit_split: str) -> tuple[int, int]:
    n_games = int(meta["n_games"])
    if audit_split == "all":
        return 0, n_games
    if audit_split not in SPLIT_ORDER:
        raise ValueError(
            f"audit_split must be one of {', '.join(AUDIT_SPLITS)}; got {audit_split!r}"
        )

    raw_ranges = meta.get("split_ranges")
    if not isinstance(raw_ranges, dict) or audit_split not in raw_ranges:
        raise ValueError(
            "Cache metadata is missing explicit split_ranges; rebuild the cache."
        )
    raw = raw_ranges[audit_split]
    if isinstance(raw, dict):
        return int(raw["start"]), int(raw["stop"])
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return int(raw[0]), int(raw[1])
    raise ValueError("Cache split range is invalid; rebuild the cache.")


def _select_audit_probabilities(
    blue_probability: np.ndarray,
    *,
    total_games: int,
    game_slice: slice,
    split_games: int,
) -> np.ndarray:
    probabilities = np.asarray(blue_probability)
    if probabilities.shape in {(total_games,), (total_games, 10)}:
        return probabilities[game_slice]
    if probabilities.shape in {(split_games,), (split_games, 10)}:
        return probabilities
    raise ValueError(
        "blue probability array must have shape [all games], [all games, 10], "
        "[split games], or [split games, 10]"
    )


@dataclass(frozen=True)
class AuditBin:
    label: str
    n: int
    empirical_wr: float
    hgnn_wr: float
    gap: float
    gap_ci95_low: float = float("nan")
    gap_ci95_high: float = float("nan")
    bootstrap_samples: int = 0
    # focus-row classification accuracy at the 0.5 threshold
    accuracy: float = float("nan")
    # accuracy after shifting this bin's predictions by -gap (mean matches the
    # empirical WR) while preserving the model's within-bin ranking
    calibrated_accuracy: float = float("nan")


@dataclass(frozen=True)
class AuditRow:
    spec: AuditSpec
    bins: tuple[AuditBin, ...]

    @property
    def endpoint_effect(self) -> float:
        populated = [row for row in self.bins if row.n > 0]
        if len(populated) < 2:
            return float("nan")
        return populated[-1].empirical_wr - populated[0].empirical_wr

    @property
    def hgnn_endpoint_effect(self) -> float:
        populated = [row for row in self.bins if row.n > 0]
        if len(populated) < 2:
            return float("nan")
        return populated[-1].hgnn_wr - populated[0].hgnn_wr

    @property
    def effect_shrinkage_ratio(self) -> float:
        return _effect_shrinkage_ratio(self.hgnn_endpoint_effect, self.endpoint_effect)

    @property
    def is_flagged(self) -> bool:
        return self.spec.title in FLAGGED_AUDIT_TITLES


@dataclass(frozen=True)
class AuditSplitSummary:
    split: str
    n_games: int
    n_tests: int
    n_populated_bins: int
    mean_abs_gap: float
    max_abs_gap: float
    gap_mse: float
    accuracy: float = float("nan")
    calibrated_accuracy: float = float("nan")
    calibration_lift: float = float("nan")

    @property
    def n_focus_rows(self) -> int:
        return self.n_games * 10


class AuditData:
    def __init__(
        self,
        *,
        context_cache_dir: Path,
        blue_probability: np.ndarray,
        audit_split: str = "all",
    ) -> None:
        self.cache_dir = context_cache_dir
        meta = json.loads(
            (context_cache_dir / "cache_meta.json").read_text(encoding="utf-8")
        )
        self.total_games = int(meta["n_games"])
        split_lo, split_hi = _audit_split_range(meta, audit_split)
        self.game_slice = slice(split_lo, split_hi)
        self.n_games = split_hi - split_lo
        if self.n_games < 0:
            raise ValueError(f"invalid audit split range for {audit_split!r}")
        selected_probability = _select_audit_probabilities(
            blue_probability,
            total_games=self.total_games,
            game_slice=self.game_slice,
            split_games=self.n_games,
        )
        if selected_probability.shape not in {(self.n_games,), (self.n_games, 10)}:
            raise ValueError(
                "blue probability array must have shape [games] or [games, 10]"
            )
        self.audit_split = audit_split
        self.build_vocab = tuple(meta["identity"]["build_vocab"])
        self.blue_win = np.load(context_cache_dir / "blue_win.npy", mmap_mode="r")[
            self.game_slice
        ]
        self.champion_id = np.load(
            context_cache_dir / "champion_id.npy", mmap_mode="r"
        )[self.game_slice]
        self.build_id = np.load(context_cache_dir / "build_id.npy", mmap_mode="r")[
            self.game_slice
        ]
        self.context_raw = np.load(
            context_cache_dir / "identity_context_raw.npy", mmap_mode="r"
        )[self.game_slice]
        self.blue_probability = np.asarray(selected_probability, dtype=np.float64)
        self._hp_lookup, self._range_lookup = _static_lookups()
        self._lens = AuditLens(
            champion_id=self.champion_id,
            build_id=self.build_id,
            context_raw=self.context_raw,
            build_vocab=self.build_vocab,
            hp_lookup=self._hp_lookup,
            range_lookup=self._range_lookup,
        )

    @property
    def labels(self) -> np.ndarray:
        return np.concatenate(
            [
                np.repeat(self.blue_win[:, None].astype(np.float64), 5, axis=1),
                np.repeat((1.0 - self.blue_win[:, None]).astype(np.float64), 5, axis=1),
            ],
            axis=1,
        )

    @property
    def predictions(self) -> np.ndarray:
        if self.blue_probability.ndim == 2:
            return self.blue_probability
        p = self.blue_probability[:, None]
        return np.concatenate(
            [
                np.repeat(p, 5, axis=1),
                np.repeat(1.0 - p, 5, axis=1),
            ],
            axis=1,
        )

    @property
    def slot_hp(self) -> np.ndarray:
        return self._lens.slot_hp

    @property
    def slot_range(self) -> np.ndarray:
        return self._lens.slot_range

    def axis(self, name: str) -> np.ndarray:
        return self._lens.axis(name)

    def focus_mask(self, spec: AuditSpec) -> np.ndarray:
        return self._lens.focus_mask(spec)


def evaluate_specs(data: AuditData, specs: Sequence[AuditSpec]) -> tuple[AuditRow, ...]:
    return evaluate_specs_with_bootstrap(
        data, specs, bootstrap_samples=0, bootstrap_seed=0
    )


def evaluate_specs_with_bootstrap(
    data: AuditData,
    specs: Sequence[AuditSpec],
    *,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 0,
) -> tuple[AuditRow, ...]:
    labels = data.labels
    predictions = data.predictions
    rows: list[AuditRow] = []
    bootstrap_samples = max(int(bootstrap_samples), 0)
    rng = np.random.default_rng(int(bootstrap_seed))
    for spec in specs:
        focus = data.focus_mask(spec)
        axis = data.axis(spec.axis)
        bins: list[AuditBin] = []
        for bin_spec in spec.bins:
            mask = focus & bin_spec.predicate(axis)
            n = int(mask.sum())
            bin_labels = labels[mask]
            bin_predictions = predictions[mask]
            empirical = _mean_or_nan(bin_labels)
            hgnn = _mean_or_nan(bin_predictions)
            gap = hgnn - empirical
            ci_low, ci_high = _bootstrap_gap_ci(
                bin_labels,
                bin_predictions,
                samples=bootstrap_samples,
                rng=rng,
            )
            correct = (bin_predictions >= 0.5).astype(np.float64) == bin_labels
            # Perfect calibration of this bin: shift predictions so the bin mean
            # matches the empirical WR, keeping the within-bin ranking, re-threshold.
            calibrated_correct = (bin_predictions - gap >= 0.5).astype(
                np.float64
            ) == bin_labels
            bins.append(
                AuditBin(
                    label=bin_spec.label,
                    n=n,
                    empirical_wr=empirical,
                    hgnn_wr=hgnn,
                    gap=gap,
                    gap_ci95_low=ci_low,
                    gap_ci95_high=ci_high,
                    bootstrap_samples=bootstrap_samples,
                    accuracy=_mean_or_nan(correct.astype(np.float64)),
                    calibrated_accuracy=_mean_or_nan(
                        calibrated_correct.astype(np.float64)
                    ),
                )
            )
        rows.append(AuditRow(spec=spec, bins=tuple(bins)))
    return tuple(rows)


def summarize_audit_split(
    *,
    context_cache_dir: Path,
    blue_probability: np.ndarray,
    audit_split: str,
    specs: Sequence[AuditSpec],
) -> AuditSplitSummary:
    data = AuditData(
        context_cache_dir=context_cache_dir,
        blue_probability=blue_probability,
        audit_split=audit_split,
    )
    rows = evaluate_specs(data, specs)
    summary = gap_summary([bin_row for row in rows for bin_row in row.bins])
    return AuditSplitSummary(
        split=audit_split,
        n_games=data.n_games,
        n_tests=len(rows),
        n_populated_bins=int(summary["n_populated_bins"]),
        mean_abs_gap=float(summary["mean_abs_gap"]),
        max_abs_gap=float(summary["max_abs_gap"]),
        gap_mse=float(summary["gap_mse"]),
        accuracy=float(summary["accuracy"]),
        calibrated_accuracy=float(summary["calibrated_accuracy"]),
        calibration_lift=float(summary["calibration_lift"]),
    )


def predict_blue_probabilities(
    *,
    model_path: Path,
    cache_dir: Path,
    encoder_sidecar_path: Path | None,
    batch_size: int,
    device: str,
) -> np.ndarray:
    device = resolve_device(device)
    model, config, strength = load_hgnn_model(model_path, device=device)
    model.eval()
    dataset_cfg = DatasetConfig(
        cache_dir=cache_dir, encoder_sidecar_path=encoder_sidecar_path
    )
    load_semantic_group_features = bool(
        config.use_learned_semantic_moe and config.use_semantic_group_features
    )
    splits = load_splits(
        dataset_cfg,
        require_counts=True,
        load_semantic_group_features=load_semantic_group_features,
        semantic_group_feature_dim=int(config.semantic_group_feature_dim),
    )
    gatherer = None
    if _model_uses_sidecar(config) and splits["train"].identity_static_sidecar is None:
        gatherer = _build_sidecar_gatherer(
            dataset_cfg,
            identity_meta(dataset_cfg),
            config,
            device=device,
        )
    outputs = [
        _predict_split(
            model,
            split,
            batch_size=batch_size,
            strength=strength,
            device=device,
            gatherer=gatherer,
        )
        for split in (splits[name] for name in SPLIT_ORDER)
    ]
    return np.concatenate(outputs).astype(np.float64)


def _predict_split(
    model: HGNNWinModel,
    split: SplitData,
    *,
    batch_size: int,
    strength: float,
    device: str,
    gatherer: _SidecarGatherer | None = None,
) -> np.ndarray:
    out: list[np.ndarray] = []
    with torch.no_grad():
        n_rows = int(split.blue_win.size)
        for start in range(0, n_rows, batch_size):
            rows = slice(start, min(start + batch_size, n_rows))
            champion_id = split.champion_id[rows]
            build_id = split.build_id[rows]
            gathered_sidecar = (
                None
                if gatherer is None or split.identity_static_sidecar is not None
                else gatherer.gather(
                    torch.as_tensor(
                        np.array(champion_id, copy=True),
                        dtype=torch.long,
                        device=device,
                    ),
                    torch.as_tensor(
                        np.array(build_id, copy=True), dtype=torch.long, device=device
                    ),
                )
            )
            if gathered_sidecar is None:
                identity_static_sidecar = (
                    None
                    if split.identity_static_sidecar is None
                    else split.identity_static_sidecar[rows]
                )
                identity_full_game_sidecar = (
                    None
                    if split.identity_full_game_sidecar is None
                    else split.identity_full_game_sidecar[rows]
                )
                identity_temporal_sidecar = (
                    None
                    if split.identity_temporal_sidecar is None
                    else split.identity_temporal_sidecar[rows]
                )
                identity_encoder_support = (
                    None
                    if split.identity_encoder_support is None
                    else split.identity_encoder_support[rows]
                )
            else:
                identity_static_sidecar = gathered_sidecar["identity_static_sidecar"]
                identity_full_game_sidecar = gathered_sidecar[
                    "identity_full_game_sidecar"
                ]
                identity_temporal_sidecar = gathered_sidecar[
                    "identity_temporal_sidecar"
                ]
                identity_encoder_support = gathered_sidecar["identity_encoder_support"]
            inputs = build_hgnn_inputs(
                champion_id=champion_id,
                build_id=build_id,
                win_rate=split.win_rate[rows],
                p1_cnt=split.p1_cnt[rows],
                strength=strength,
                identity_static_sidecar=identity_static_sidecar,
                identity_full_game_sidecar=identity_full_game_sidecar,
                identity_temporal_sidecar=identity_temporal_sidecar,
                identity_encoder_support=identity_encoder_support,
                semantic_group_features=(
                    None
                    if split.semantic_group_features is None
                    else split.semantic_group_features[rows]
                ),
                loadout_features=(
                    None
                    if split.loadout_features is None
                    else split.loadout_features[rows]
                ),
                patch_features=(
                    None if split.patch_features is None else split.patch_features[rows]
                ),
                device=device,
            )
            outputs = model(**inputs)
            focus_probabilities = _focus_side_probabilities_from_outputs(outputs)
            out.append(focus_probabilities.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def _focus_side_probabilities_from_outputs(
    outputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    slot_delta = outputs.get("semantic_moe_slot_delta")
    if slot_delta is None:
        blue = torch.sigmoid(outputs["final_logit"]).view(-1, 1)
        return torch.cat([blue.expand(-1, 5), (1.0 - blue).expand(-1, 5)], dim=1)

    base_logit = outputs["base_logit"]
    context_logit = outputs["context_logit"]
    semantic_moe_logit = outputs.get("semantic_moe_logit")
    if semantic_moe_logit is None:
        semantic_moe_logit = base_logit.new_zeros(base_logit.shape)
    feature_logit = outputs.get("feature_logit")
    if feature_logit is None:
        feature_logit = base_logit.new_zeros(base_logit.shape)
    shared_logit = base_logit + context_logit - semantic_moe_logit + feature_logit
    blue_delta = slot_delta[:, :5]
    red_delta = slot_delta[:, 5:]
    blue_focus_logit = (
        shared_logit[:, None]
        + blue_delta
        - red_delta.mean(
            dim=1,
            keepdim=True,
        )
    )
    red_focus_logit = (
        -shared_logit[:, None]
        + red_delta
        - blue_delta.mean(
            dim=1,
            keepdim=True,
        )
    )
    return torch.cat(
        [torch.sigmoid(blue_focus_logit), torch.sigmoid(red_focus_logit)],
        dim=1,
    )


def load_or_predict_blue_probabilities(
    *,
    model_path: Path,
    model_cache_dir: Path,
    encoder_sidecar_path: Path | None,
    prediction_cache: Path,
    n_games: int,
    refresh: bool,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if not refresh and prediction_cache.exists():
        cached = np.load(prediction_cache)
        if cached.shape in {(n_games,), (n_games, 10)}:
            return np.asarray(cached, dtype=np.float64)
    probabilities = predict_blue_probabilities(
        model_path=model_path,
        cache_dir=model_cache_dir,
        encoder_sidecar_path=encoder_sidecar_path,
        batch_size=batch_size,
        device=device,
    )
    if probabilities.shape not in {(n_games,), (n_games, 10)}:
        raise ValueError(
            "predicted probability count does not match context cache n_games"
        )
    prediction_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(prediction_cache, probabilities.astype(np.float32))
    return probabilities


def render_audit(
    rows: Sequence[AuditRow],
    *,
    model_path: Path,
    model_cache_dir: Path,
    context_cache_dir: Path,
    encoder_sidecar_path: Path | None = None,
    prediction_cache: Path | None = None,
    audit_split: str = "all",
    audited_games: int | None = None,
    split_summaries: Sequence[AuditSplitSummary] = (),
    updated: str | None = None,
) -> str:
    updated = updated or date.today().isoformat()
    by_section: dict[str, list[AuditRow]] = {}
    for row in rows:
        by_section.setdefault(row.spec.section, []).append(row)
    split_description = (
        "all splits combined" if audit_split == "all" else f"`{audit_split}` split only"
    )
    lines = [
        "# HGNN Context Examples Audit",
        "",
        f"Updated: {updated}.",
        "",
        "This audit joins the empirical focus-side context examples to the trained "
        "semantic HGNN predictions for the same cached games. Each audit is its own "
        "table: one row per threshold bin reporting `n / empirical WR / HGNN WR / gap "
        "/ accuracy`, with a per-table Gap MSE, accuracy, and the accuracy headroom "
        "from perfect calibration (`Calibration lift`) above it. Gap is "
        "`HGNN WR - empirical WR`; zero gap is the target.",
        "",
        "## Scope And Threshold Definitions",
        "",
        f"- Context source: `{context_cache_dir}` side-row arrays, {split_description}.",
        f"- HGNN model: `{model_path}`.",
        f"- HGNN cache: `{model_cache_dir}`.",
        (
            f"- Encoder sidecar artifact: `{encoder_sidecar_path}`."
            if encoder_sidecar_path is not None
            else "- Encoder sidecar artifact: cache metadata or materialized cache arrays."
        ),
        "- HGNN WR uses focus-slot semantic MoE probabilities when a checkpoint exposes slot deltas; older checkpoints fall back to raw `final_logit` probabilities.",
        f"- Semantic group feature schema: v{SEMANTIC_GROUP_FEATURE_SCHEMA_VERSION}, {SEMANTIC_GROUP_FEATURE_DIM} compact per-slot features; used only by checkpoints trained with `--use-semantic-group-features`.",
        *(
            [
                f"- Games audited: {audited_games:,}.",
                f"- Focus-slot rows audited: {audited_games * 10:,}.",
            ]
            if audited_games is not None
            else []
        ),
        "- Model-alignment rows score each slot with its focus-side win probability; blue-side slots use the blue-team frame and red-side slots use the mirrored red-team frame.",
        "- Continuous thresholds are global side-row team-average percentiles.",
        "- Count thresholds use explicit enemy-team counts.",
        "- WR, effects, and gaps are focus-side win-rate percentage points.",
        "- Accuracy is focus-row classification accuracy at the 0.5 threshold (HGNN focus WR >= 0.5 predicts a focus-side win); the per-table value is bin-n weighted.",
        "- `Acc if calibrated` shifts each bin's predictions so the bin mean equals the empirical WR (perfect calibration) while keeping the model's within-bin ranking, then re-thresholds at 0.5; `Calibration lift` is that minus current accuracy -- the true accuracy impact of closing the gap. It is near zero because accuracy is limited by ranking, not calibration.",
        "- Selected-enchanter probe uses Sona, Karma, Lulu, and Zilean in `UTILITY` with `utility_enchanter` or `utility_protection`.",
        "- Low own-damage probe is anchored once per team side, then compared against the enemy heal/shield context.",
        "- Effect shrinkage is `HGNN effect / empirical effect`; values below 1.0 mean the model under-expresses the observed context effect.",
        "",
        "| Axis | Low threshold | High threshold | Notes |",
        "|---|---|---|---|",
        "| Physical share | `<= 0.387` | `>= 0.557` | Team-average identity-context physical share. |",
        "| Magic share | `<= 0.373` | `>= 0.549` | Team-average identity-context magic share. |",
        "| Damage pressure | `<= 0.739` | `>= 0.813` | Team-average champion damage pressure. |",
        "| Damage-taken pressure | `<= 0.639` | `>= 0.721` | Team-average damage-taken pressure. |",
        "| Heal/shield pressure | `<= 0.028` | `>= 0.202` | Team-average ally heal/shield pressure. |",
        "| CC pressure | `<= 0.374` | `>= 0.539` | Team-average crowd-control pressure. |",
        "| Siege pressure | `<= 0.441` | `>= 0.530` | Team-average siege and structure pressure. |",
        "| Scaling pressure | `<= 0.829` | `>= 0.863` | Team-average scaling pressure. |",
        f"| Burst-proxy count | `0` | `>= 3` | Enemy slots with slot damage pressure `>= {BURST_DAMAGE_THRESHOLD:.3f}` and a non-tank build. |",
        f"| Hard-CC count | `0` | `>= 3` | Enemy slots with slot CC pressure `>= {HARD_CC_THRESHOLD:.3f}`. |",
        "| Tank/frontline count | `0` | `>= 3` | Enemy builds in `ar_tank`, `mr_tank`, `ad_off_tank`, or `ap_off_tank`. |",
        f"| Heavy damage-taken count | `0` | `>= 3` | Enemy slots with slot damage-taken pressure `>= {HEAVY_TAKEN_THRESHOLD:.3f}`. |",
        f"| High-HP count | `0` | `>= 3` | Enemy champions with static level-18 HP `>= {HIGH_HP_THRESHOLD:.1f}`. |",
        f"| Focus HP tier | `<= {FOCUS_HP_LOW_THRESHOLD:.1f}` | `>= {HIGH_HP_THRESHOLD:.1f}` | Static champion level-18 HP. |",
        f"| Ranged count | `<= 1` | `>= 4` | Static `attackRange_flat > {RANGED_ATTACK_RANGE_THRESHOLD:.0f}` as ranged. |",
        f"| Same-role range | `<= {RANGED_ATTACK_RANGE_THRESHOLD:.0f}` | `> {RANGED_ATTACK_RANGE_THRESHOLD:.0f}` | Static attack range for the lane opponent. |",
        "| Skirmish-ally count | `0` | `>= 2` | Gwen, Jax, Irelia, Fiora, Udyr, and XinZhao on the focus team. |",
        "",
        "## Gap Summary",
        "",
        "| Section | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for section, section_rows in by_section.items():
        summary = gap_summary([bin_row for row in section_rows for bin_row in row.bins])
        lines.append(
            "| "
            + " | ".join(
                [
                    section,
                    str(len(section_rows)),
                    str(summary["n_populated_bins"]),
                    _format_pp(summary["mean_abs_gap"], signed=False),
                    _format_pp(summary["max_abs_gap"], signed=False),
                    _format_pp_mse(summary["gap_mse"]),
                    _format_pct(summary["accuracy"]),
                    _format_pct(summary["calibrated_accuracy"]),
                    _format_pp(summary["calibration_lift"]),
                ]
            )
            + " |"
        )
    if split_summaries:
        lines.extend(
            [
                "",
                "## Train And Test Summary",
                "",
                "These rows reuse the same audit specs and prediction cache, but evaluate "
                "the cached train and test ranges separately.",
                "",
                "| Split | Games | Focus-slot rows | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for summary in split_summaries:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_split_label(summary.split),
                        f"{summary.n_games:,}",
                        f"{summary.n_focus_rows:,}",
                        str(summary.n_tests),
                        str(summary.n_populated_bins),
                        _format_pp(summary.mean_abs_gap, signed=False),
                        _format_pp(summary.max_abs_gap, signed=False),
                        _format_pp_mse(summary.gap_mse),
                        _format_pct(summary.accuracy),
                        _format_pct(summary.calibrated_accuracy),
                        _format_pp(summary.calibration_lift),
                    ]
                )
                + " |"
            )
    tail_rows = [row for row in rows if _is_enemy_count_axis(row.spec.axis)]
    if tail_rows:
        lines.extend(
            [
                "",
                "## Enemy Count Tail Shrinkage",
                "",
                "| Audit | Axis | Baseline bin | Tail bin | Empirical tail effect | HGNN tail effect | Shrinkage |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in tail_rows:
            first, last = _declared_endpoint_bins(row.bins)
            endpoints_populated = (
                first is not None
                and last is not None
                and first is not last
                and first.n > 0
                and last.n > 0
            )
            if endpoints_populated:
                empirical_effect = last.empirical_wr - first.empirical_wr
                hgnn_effect = last.hgnn_wr - first.hgnn_wr
                shrinkage = _effect_shrinkage_ratio(hgnn_effect, empirical_effect)
            else:
                empirical_effect = float("nan")
                hgnn_effect = float("nan")
                shrinkage = float("nan")
            lines.append(
                "| "
                + " | ".join(
                    [
                        row.spec.title,
                        f"`{row.spec.axis}`",
                        _format_endpoint_bin(first),
                        _format_endpoint_bin(last),
                        _format_pp(empirical_effect),
                        _format_pp(hgnn_effect),
                        _format_ratio(shrinkage),
                    ]
                )
                + " |"
            )
    for section, section_rows in by_section.items():
        lines.extend(["", f"## {section}", ""])
        for row in section_rows:
            lines.extend(_format_audit_table(row))
    all_bins = [bin_row for row in rows for bin_row in row.bins]
    summary = gap_summary(all_bins)
    lines.extend(
        [
            "",
            "## Overall Summary",
            "",
            f"Detailed audit tables above are rendered from the `{audit_split}` split.",
            "",
            "| Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| "
            + " | ".join(
                [
                    str(len(rows)),
                    str(summary["n_populated_bins"]),
                    _format_pp(summary["mean_abs_gap"], signed=False),
                    _format_pp(summary["max_abs_gap"], signed=False),
                    _format_pp_mse(summary["gap_mse"]),
                    _format_pct(summary["accuracy"]),
                    _format_pct(summary["calibrated_accuracy"]),
                    _format_pp(summary["calibration_lift"]),
                ]
            )
            + " |",
        ]
    )
    if split_summaries:
        lines.extend(
            [
                "",
                "| Split | Games | Focus-slot rows | Tests | Populated bins | Mean abs gap | Max abs gap | Gap MSE | Accuracy | Acc if calibrated | Calibration lift |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for split_summary in split_summaries:
            lines.append(
                "| "
                + " | ".join(
                    [
                        _format_split_label(split_summary.split),
                        f"{split_summary.n_games:,}",
                        f"{split_summary.n_focus_rows:,}",
                        str(split_summary.n_tests),
                        str(split_summary.n_populated_bins),
                        _format_pp(split_summary.mean_abs_gap, signed=False),
                        _format_pp(split_summary.max_abs_gap, signed=False),
                        _format_pp_mse(split_summary.gap_mse),
                        _format_pct(split_summary.accuracy),
                        _format_pct(split_summary.calibrated_accuracy),
                        _format_pp(split_summary.calibration_lift),
                    ]
                )
                + " |"
            )
    lines.extend(
        [
            "",
            "Gap MSE is `mean((HGNN_focus_WR - empirical_focus_WR)^2)` across populated "
            "threshold bins, rendered as percentage-points squared.",
            "",
            "## Reproduction Commands",
            "",
            "The checked-in report uses the focus-slot audit path. Checkpoints with semantic "
            "MoE slot deltas are scored with per-slot focus-side probabilities instead of one "
            "repeated match-level probability. Regenerate predictions from the selected "
            "checkpoint with `--refresh-predictions`; omit it to reuse the prediction cache "
            "for report-only updates.",
            "",
            "```bash",
            "uv run python -m app.ml.context_examples_audit \\",
            f"  --context-cache-dir {context_cache_dir} \\",
            f"  --model-cache-dir {model_cache_dir} \\",
            f"  --model-path {model_path} \\",
            *(
                [f"  --encoder-sidecar-path {encoder_sidecar_path} \\"]
                if encoder_sidecar_path is not None
                else []
            ),
            *(
                [f"  --prediction-cache {prediction_cache} \\"]
                if prediction_cache is not None
                else []
            ),
            f"  --audit-split {audit_split} \\",
            f"  --output {DEFAULT_OUTPUT_PATH} \\",
            "  --refresh-predictions",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit(path: Path, markdown: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")


def gap_summary(rows: Sequence[AuditBin]) -> dict[str, float | int]:
    populated = [row for row in rows if row.n > 0 and np.isfinite(row.gap)]
    gaps = np.asarray([row.gap for row in populated], dtype=np.float64)
    counts = np.asarray([row.n for row in populated], dtype=np.float64)
    accuracies = np.asarray([row.accuracy for row in populated], dtype=np.float64)
    calibrated = np.asarray(
        [row.calibrated_accuracy for row in populated], dtype=np.float64
    )
    total_n = float(counts.sum())
    accuracy = (
        float(np.sum(counts * accuracies) / total_n) if total_n > 0 else float("nan")
    )
    # Accuracy if every bin were perfectly calibrated (mean shifted to the empirical
    # WR) while keeping the model's within-bin ranking -- the true accuracy impact.
    calibrated_accuracy = (
        float(np.sum(counts * calibrated) / total_n) if total_n > 0 else float("nan")
    )
    return {
        "n_populated_bins": int(sum(row.n > 0 for row in rows)),
        "n_focus_rows": int(total_n),
        "mean_abs_gap": _mean_or_nan(np.abs(gaps)),
        "max_abs_gap": _max_or_nan(np.abs(gaps)),
        "gap_mse": _mean_or_nan(gaps**2),
        "support_weighted_mean_abs_gap": (
            float(np.sum(counts * np.abs(gaps)) / total_n)
            if total_n > 0
            else float("nan")
        ),
        "support_weighted_gap_mse": (
            float(np.sum(counts * (gaps**2)) / total_n) if total_n > 0 else float("nan")
        ),
        "accuracy": accuracy,
        "calibrated_accuracy": calibrated_accuracy,
        "calibration_lift": calibrated_accuracy - accuracy,
    }


def audit_json_payload(
    *,
    rows_by_split: dict[str, Sequence[AuditRow]],
    split_summaries: Sequence[AuditSplitSummary],
    model_path: Path,
    model_cache_dir: Path,
    context_cache_dir: Path,
    encoder_sidecar_path: Path | None,
    prediction_cache: Path | None,
    audit_split: str,
    updated: str | None = None,
) -> dict[str, object]:
    """Build a machine-readable companion payload for audit outcome reporting."""

    updated = updated or date.today().isoformat()
    payload = {
        "schema_version": 1,
        "updated": updated,
        "audit_split": audit_split,
        "model_path": str(model_path),
        "model_cache_dir": str(model_cache_dir),
        "context_cache_dir": str(context_cache_dir),
        "encoder_sidecar_path": (
            None if encoder_sidecar_path is None else str(encoder_sidecar_path)
        ),
        "prediction_cache": None if prediction_cache is None else str(prediction_cache),
        "flagged_audit_titles": sorted(FLAGGED_AUDIT_TITLES),
        "splits": {},
        "split_summaries": [
            _split_summary_payload(summary) for summary in split_summaries
        ],
    }
    split_payloads: dict[str, object] = {}
    for split, rows in rows_by_split.items():
        row_payloads = [_row_payload(row) for row in rows]
        split_payloads[split] = {
            "summary": _gap_summary_payload(
                gap_summary([bin_row for row in rows for bin_row in row.bins])
            ),
            "flagged_summary": _gap_summary_payload(
                gap_summary(
                    [bin_row for row in rows if row.is_flagged for bin_row in row.bins]
                )
            ),
            "rows": row_payloads,
        }
    payload["splits"] = split_payloads
    return _json_safe(payload)


def write_audit_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _row_payload(row: AuditRow) -> dict[str, object]:
    summary = gap_summary(row.bins)
    first, last = _declared_endpoint_bins(row.bins)
    declared_empirical_effect = (
        last.empirical_wr - first.empirical_wr
        if first is not None and last is not None and first.n > 0 and last.n > 0
        else float("nan")
    )
    declared_hgnn_effect = (
        last.hgnn_wr - first.hgnn_wr
        if first is not None and last is not None and first.n > 0 and last.n > 0
        else float("nan")
    )
    diagnostics = _row_diagnostics(
        row.bins,
        endpoint_effect=row.endpoint_effect,
        hgnn_endpoint_effect=row.hgnn_endpoint_effect,
    )
    return {
        "section": row.spec.section,
        "title": row.spec.title,
        "read": row.spec.read,
        "axis": row.spec.axis,
        "champions": list(row.spec.champions),
        "positions": list(row.spec.positions),
        "builds": list(row.spec.builds),
        "focus_condition": row.spec.focus_condition,
        "is_flagged": row.is_flagged,
        "summary": _gap_summary_payload(summary),
        "endpoint_effect": row.endpoint_effect,
        "hgnn_endpoint_effect": row.hgnn_endpoint_effect,
        "effect_shrinkage_ratio": row.effect_shrinkage_ratio,
        "declared_endpoint_effect": declared_empirical_effect,
        "declared_hgnn_endpoint_effect": declared_hgnn_effect,
        "declared_effect_shrinkage_ratio": _effect_shrinkage_ratio(
            declared_hgnn_effect,
            declared_empirical_effect,
        ),
        "diagnostics": diagnostics,
        **diagnostics,
        "bins": [_bin_payload(bin_row) for bin_row in row.bins],
    }


def _row_diagnostics(
    rows: Sequence[AuditBin],
    *,
    endpoint_effect: float,
    hgnn_endpoint_effect: float,
) -> dict[str, object]:
    """Decompose a context row into level, slope, and tail calibration signals."""
    populated = [row for row in rows if row.n > 0 and math.isfinite(row.gap)]
    if not populated:
        return {
            "level_gap": float("nan"),
            "slope_gap": float("nan"),
            "tail_gap": float("nan"),
            "tail_bin_label": None,
            "tail_bin_n": 0,
            "tail_empirical_wr": float("nan"),
            "tail_hgnn_wr": float("nan"),
            "tail_gap_ci95_low": float("nan"),
            "tail_gap_ci95_high": float("nan"),
            "direction_correct": None,
        }

    counts = np.asarray([row.n for row in populated], dtype=np.float64)
    gaps = np.asarray([row.gap for row in populated], dtype=np.float64)
    total_n = float(counts.sum())
    level_gap = float(np.sum(counts * gaps) / total_n) if total_n > 0 else float("nan")
    slope_gap = (
        float(hgnn_endpoint_effect - endpoint_effect)
        if math.isfinite(hgnn_endpoint_effect) and math.isfinite(endpoint_effect)
        else float("nan")
    )
    direction_correct: bool | None
    if not math.isfinite(hgnn_endpoint_effect) or not math.isfinite(endpoint_effect):
        direction_correct = None
    elif abs(endpoint_effect) < 1.0e-12:
        direction_correct = None
    else:
        direction_correct = (hgnn_endpoint_effect == 0.0) or (
            math.copysign(1.0, hgnn_endpoint_effect)
            == math.copysign(1.0, endpoint_effect)
        )

    tail = populated[-1]
    return {
        "level_gap": level_gap,
        "slope_gap": slope_gap,
        "tail_gap": tail.gap,
        "tail_bin_label": tail.label,
        "tail_bin_n": int(tail.n),
        "tail_empirical_wr": tail.empirical_wr,
        "tail_hgnn_wr": tail.hgnn_wr,
        "tail_gap_ci95_low": tail.gap_ci95_low,
        "tail_gap_ci95_high": tail.gap_ci95_high,
        "direction_correct": direction_correct,
    }


def _bin_payload(row: AuditBin) -> dict[str, object]:
    return {
        "label": row.label,
        "n": int(row.n),
        "empirical_wr": row.empirical_wr,
        "hgnn_wr": row.hgnn_wr,
        "gap": row.gap,
        "gap_ci95_low": row.gap_ci95_low,
        "gap_ci95_high": row.gap_ci95_high,
        "bootstrap_samples": int(row.bootstrap_samples),
        "accuracy": row.accuracy,
        "calibrated_accuracy": row.calibrated_accuracy,
        "calibration_lift": row.calibrated_accuracy - row.accuracy,
    }


def _split_summary_payload(summary: AuditSplitSummary) -> dict[str, object]:
    return {
        "split": summary.split,
        "n_games": int(summary.n_games),
        "n_focus_rows": int(summary.n_focus_rows),
        "n_tests": int(summary.n_tests),
        "n_populated_bins": int(summary.n_populated_bins),
        "mean_abs_gap": summary.mean_abs_gap,
        "max_abs_gap": summary.max_abs_gap,
        "gap_mse": summary.gap_mse,
        "accuracy": summary.accuracy,
        "calibrated_accuracy": summary.calibrated_accuracy,
        "calibration_lift": summary.calibration_lift,
    }


def _gap_summary_payload(summary: dict[str, float | int]) -> dict[str, object]:
    return {key: value for key, value in summary.items()}


def _bootstrap_gap_ci(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    n = int(labels.shape[0])
    if samples <= 0 or n <= 0:
        return float("nan"), float("nan")
    if n == 1:
        gap = float(predictions.mean() - labels.mean())
        return gap, gap
    gaps = np.empty(samples, dtype=np.float64)
    for idx in range(samples):
        sample = rng.integers(0, n, size=n)
        gaps[idx] = float(predictions[sample].mean() - labels[sample].mean())
    return (
        float(np.percentile(gaps, 2.5)),
        float(np.percentile(gaps, 97.5)),
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _effect_shrinkage_ratio(hgnn_effect: float, empirical_effect: float) -> float:
    if not math.isfinite(hgnn_effect) or not math.isfinite(empirical_effect):
        return float("nan")
    if abs(empirical_effect) < 1.0e-12:
        return float("nan")
    return float(hgnn_effect / empirical_effect)


def _declared_endpoint_bins(
    rows: Sequence[AuditBin],
) -> tuple[AuditBin | None, AuditBin | None]:
    if not rows:
        return None, None
    return rows[0], rows[-1]


def _format_endpoint_bin(row: AuditBin | None) -> str:
    if row is None:
        return "N/A (empty)"
    suffix = " (empty)" if row.n <= 0 else ""
    return f"`{row.label}`{suffix}"


def _is_enemy_count_axis(axis: str) -> bool:
    return axis.startswith("enemy_") and axis.endswith("_count")


def _static_lookups() -> tuple[np.ndarray, np.ndarray]:
    return static_hp_range_lookups()


def _format_split_label(split: str) -> str:
    if split == "all":
        return "All"
    return split.capitalize()


def _format_ratio(value: float) -> str:
    if not math.isfinite(value):
        return "N/A"
    return f"{value:.2f}x"


def _format_audit_table(row: AuditRow) -> list[str]:
    summary = gap_summary(list(row.bins))
    stats = " | ".join(
        [
            f"**Gap MSE** {_format_pp_mse(summary['gap_mse'])}",
            f"**Mean abs gap** {_format_pp(summary['mean_abs_gap'], signed=False)}",
            f"**Accuracy** {_format_pct(summary['accuracy'])}",
            f"**Accuracy if calibrated** {_format_pct(summary['calibrated_accuracy'])}",
            f"**Calibration lift** {_format_pp(summary['calibration_lift'])}",
            f"**Empirical effect** {_format_pp(row.endpoint_effect)}",
            f"**HGNN effect** {_format_pp(row.hgnn_endpoint_effect)}",
            f"**Shrinkage** {_format_ratio(row.effect_shrinkage_ratio)}",
        ]
    )
    lines = [
        f"### {row.spec.title}",
        "",
        row.spec.read,
        "",
        stats,
        "",
        "| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bin_row in row.bins:
        if bin_row.n <= 0:
            lines.append(f"| `{bin_row.label}` | 0 | N/A | N/A | N/A | N/A |")
            continue
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{bin_row.label}`",
                    f"{bin_row.n:,}",
                    _format_pct(bin_row.empirical_wr),
                    _format_pct(bin_row.hgnn_wr),
                    _format_pp(bin_row.gap),
                    _format_pct(bin_row.accuracy),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--context-cache-dir", type=Path, default=DEFAULT_CONTEXT_CACHE_DIR
    )
    parser.add_argument("--model-cache-dir", type=Path, default=DEFAULT_MODEL_CACHE_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--encoder-sidecar-path", type=Path, default=None)
    parser.add_argument(
        "--prediction-cache", type=Path, default=DEFAULT_PREDICTION_CACHE
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional machine-readable payload for audit outcome reports.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Per-bin bootstrap resamples for JSON gap CIs. Default writes null CI fields.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260604)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--audit-split",
        choices=AUDIT_SPLITS,
        default="all",
        help="Cache split to audit. The default keeps the historical all-split report.",
    )
    parser.add_argument("--refresh-predictions", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    meta = json.loads(
        (args.context_cache_dir / "cache_meta.json").read_text(encoding="utf-8")
    )
    n_games = int(meta["n_games"])
    probabilities = load_or_predict_blue_probabilities(
        model_path=args.model_path,
        model_cache_dir=args.model_cache_dir,
        encoder_sidecar_path=args.encoder_sidecar_path,
        prediction_cache=args.prediction_cache,
        n_games=n_games,
        refresh=bool(args.refresh_predictions),
        batch_size=int(args.batch_size),
        device=str(args.device),
    )
    specs = audit_specs()
    data = AuditData(
        context_cache_dir=args.context_cache_dir,
        blue_probability=probabilities,
        audit_split=str(args.audit_split),
    )
    rows = evaluate_specs_with_bootstrap(
        data,
        specs,
        bootstrap_samples=int(args.bootstrap_samples),
        bootstrap_seed=int(args.bootstrap_seed),
    )
    split_summaries = tuple(
        summarize_audit_split(
            context_cache_dir=args.context_cache_dir,
            blue_probability=probabilities,
            audit_split=split_name,
            specs=specs,
        )
        for split_name in SPLIT_ORDER
    )
    markdown = render_audit(
        rows,
        model_path=args.model_path,
        model_cache_dir=args.model_cache_dir,
        context_cache_dir=args.context_cache_dir,
        encoder_sidecar_path=args.encoder_sidecar_path,
        prediction_cache=args.prediction_cache,
        audit_split=str(args.audit_split),
        audited_games=data.n_games,
        split_summaries=split_summaries,
    )
    write_audit(args.output, markdown)
    if args.json_output is not None:
        rows_by_split: dict[str, Sequence[AuditRow]] = {str(args.audit_split): rows}
        for split_name in SPLIT_ORDER:
            if split_name == args.audit_split:
                continue
            split_data = AuditData(
                context_cache_dir=args.context_cache_dir,
                blue_probability=probabilities,
                audit_split=split_name,
            )
            rows_by_split[split_name] = evaluate_specs_with_bootstrap(
                split_data,
                specs,
                bootstrap_samples=int(args.bootstrap_samples),
                bootstrap_seed=int(args.bootstrap_seed),
            )
        payload = audit_json_payload(
            rows_by_split=rows_by_split,
            split_summaries=split_summaries,
            model_path=args.model_path,
            model_cache_dir=args.model_cache_dir,
            context_cache_dir=args.context_cache_dir,
            encoder_sidecar_path=args.encoder_sidecar_path,
            prediction_cache=args.prediction_cache,
            audit_split=str(args.audit_split),
        )
        write_audit_json(args.json_output, payload)


if __name__ == "__main__":
    main()
