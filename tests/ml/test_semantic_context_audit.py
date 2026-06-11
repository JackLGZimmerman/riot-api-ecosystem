from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from app.ml.context_examples_audit import (
    AuditData,
    AuditBin,
    AuditRow,
    AuditSplitSummary,
    AuditSpec,
    FLAGGED_AUDIT_TITLES,
    audit_json_payload,
    gap_summary as context_gap_summary,
    _focus_side_probabilities_from_outputs,
    _predict_split,
    render_audit,
)
from app.ml.dataset import SplitData
from app.ml.semantic_context_audit import (
    ThresholdBin,
    evaluate_threshold_bins,
    gap_summary,
    render_model_alignment_audit,
    side_row_focus_probabilities,
)
from app.ml.train import (
    _focus_side_probabilities_from_outputs as _train_focus_side_probabilities_from_outputs,
)


def test_side_row_focus_probabilities_mirror_red_side_predictions() -> None:
    side_rows = side_row_focus_probabilities(
        blue_win=np.array([1.0, 1.0]),
        base_blue_probability=np.array([0.8, 0.4]),
        final_blue_probability=np.array([1.0, 0.8]),
    )

    assert side_rows["label"].shape == (20,)
    assert side_rows["base_prediction"].shape == (20,)
    assert side_rows["label"][:10].tolist() == [1.0] * 5 + [0.0] * 5
    assert np.allclose(side_rows["base_prediction"][:10], [0.8] * 5 + [0.2] * 5)
    assert np.allclose(side_rows["final_prediction"][10:], [0.8] * 5 + [0.2] * 5)


def test_threshold_audit_computes_empirical_predictions_and_gaps() -> None:
    side_rows = side_row_focus_probabilities(
        blue_win=np.array([1.0, 1.0]),
        base_blue_probability=np.array([0.8, 0.4]),
        final_blue_probability=np.array([1.0, 0.8]),
    )
    axis_values = np.tile(np.array([0.0] * 5 + [1.0] * 5), 2)
    rows = evaluate_threshold_bins(
        axis_values=axis_values,
        labels=side_rows["label"],
        base_predictions=side_rows["base_prediction"],
        final_predictions=side_rows["final_prediction"],
        bins=[
            ThresholdBin("focus blue", lower=0.0, upper=0.0),
            ThresholdBin("focus red", lower=1.0, upper=1.0),
        ],
    )

    assert rows[0].n == 10
    assert rows[0].empirical_wr == pytest.approx(1.0)
    assert rows[0].base_pred_wr == pytest.approx(0.6)
    assert rows[0].final_pred_wr == pytest.approx(0.9)
    assert rows[0].base_gap == pytest.approx(-0.4)
    assert rows[0].final_gap == pytest.approx(-0.1)
    assert rows[1].empirical_wr == pytest.approx(0.0)
    assert rows[1].base_gap == pytest.approx(0.4)
    assert rows[1].final_gap == pytest.approx(0.1)

    summary = gap_summary(rows)
    assert summary["base_mean_abs_gap"] == pytest.approx(0.4)
    assert summary["base_max_abs_gap"] == pytest.approx(0.4)
    assert summary["base_gap_mse"] == pytest.approx(0.16)
    assert summary["final_mean_abs_gap"] == pytest.approx(0.1)
    assert summary["final_max_abs_gap"] == pytest.approx(0.1)
    assert summary["final_gap_mse"] == pytest.approx(0.01)


def test_model_alignment_audit_markdown_includes_model_columns() -> None:
    rows = [
        evaluate_threshold_bins(
            axis_values=np.array([0.0, 0.0]),
            labels=np.array([1.0, 0.0]),
            base_predictions=np.array([0.8, 0.4]),
            final_predictions=np.array([0.9, 0.2]),
            bins=[ThresholdBin("all", lower=0.0, upper=0.0)],
        )[0]
    ]

    markdown = render_model_alignment_audit({"Synthetic": rows}, updated="2026-06-03")

    assert "Base predicted WR" in markdown
    assert "Final predicted WR" in markdown
    assert "Final gap MSE" in markdown
    assert "Overall Summary" in markdown
    assert "Final gap" in markdown
    assert "Synthetic" in markdown
    assert "+5.00 pp" in markdown
    assert "25.00 pp^2" in markdown


def test_context_examples_audit_render_includes_effect_and_tail_shrinkage() -> None:
    row = AuditRow(
        spec=AuditSpec(
            section="Synthetic counts",
            title="Enemy burst count",
            read="Tail should express the empirical effect.",
            axis="enemy_burst_count",
            bins=(),
        ),
        bins=(
            AuditBin("0", n=10, empirical_wr=0.40, hgnn_wr=0.45, gap=0.05),
            AuditBin("1", n=8, empirical_wr=0.50, hgnn_wr=0.52, gap=0.02),
            AuditBin(">= 3", n=6, empirical_wr=0.70, hgnn_wr=0.60, gap=-0.10),
        ),
    )
    empty_tail_row = AuditRow(
        spec=AuditSpec(
            section="Synthetic counts",
            title="Enemy hard CC count",
            read="Empty declared tail should not fall back to count two.",
            axis="enemy_hard_cc_count",
            bins=(),
        ),
        bins=(
            AuditBin("0", n=10, empirical_wr=0.40, hgnn_wr=0.45, gap=0.05),
            AuditBin("1", n=8, empirical_wr=0.50, hgnn_wr=0.52, gap=0.02),
            AuditBin("2", n=5, empirical_wr=0.60, hgnn_wr=0.55, gap=-0.05),
            AuditBin(">= 3", n=0, empirical_wr=float("nan"), hgnn_wr=float("nan"), gap=float("nan")),
        ),
    )

    markdown = render_audit(
        [row, empty_tail_row],
        model_path="model.pt",
        model_cache_dir="model-cache",
        context_cache_dir="context-cache",
        split_summaries=(
            AuditSplitSummary(
                split="train",
                n_games=100,
                n_tests=2,
                n_populated_bins=7,
                mean_abs_gap=0.01,
                max_abs_gap=0.04,
                gap_mse=0.0004,
            ),
            AuditSplitSummary(
                split="test",
                n_games=10,
                n_tests=2,
                n_populated_bins=5,
                mean_abs_gap=0.03,
                max_abs_gap=0.06,
                gap_mse=0.0016,
            ),
        ),
        updated="2026-06-03",
    )

    assert "Effect shrinkage is" in markdown
    assert "### Enemy burst count" in markdown
    assert "| Bin | n | Empirical WR | HGNN WR | Gap | Accuracy |" in markdown
    assert "**Gap MSE**" in markdown
    assert "**Accuracy**" in markdown
    assert "**Accuracy if calibrated**" in markdown
    assert "**Calibration lift**" in markdown
    assert "## Train And Test Summary" in markdown
    assert "| Train | 100 | 1,000 | 2 | 7 | 1.00 pp | 4.00 pp | 4.00 pp^2 |" in markdown
    assert "| Test | 10 | 100 | 2 | 5 | 3.00 pp | 6.00 pp | 16.00 pp^2 |" in markdown
    assert "## Enemy Count Tail Shrinkage" in markdown
    assert "Empirical tail effect" in markdown
    assert "HGNN tail effect" in markdown
    assert "Empirical effect" in markdown
    assert "HGNN effect" in markdown
    assert "Shrinkage" in markdown
    assert "50.00%" in markdown
    assert (
        "| Enemy hard CC count | `enemy_hard_cc_count` | `0` | `>= 3` (empty) | "
        "N/A | N/A | N/A |"
    ) in markdown
    assert "| Enemy hard CC count | `enemy_hard_cc_count` | `0` | `2` |" not in markdown


def test_context_examples_audit_json_tags_flagged_rows_and_weighted_gaps() -> None:
    title = next(iter(FLAGGED_AUDIT_TITLES))
    row = AuditRow(
        spec=AuditSpec(
            section="Flagged",
            title=title,
            read="Smoke-test row.",
            axis="enemy_hard_cc_count",
            bins=(),
        ),
        bins=(
            AuditBin(
                "0",
                n=10,
                empirical_wr=0.40,
                hgnn_wr=0.50,
                gap=0.10,
                gap_ci95_low=0.01,
                gap_ci95_high=0.20,
                bootstrap_samples=32,
            ),
            AuditBin(
                "empty",
                n=0,
                empirical_wr=float("nan"),
                hgnn_wr=float("nan"),
                gap=float("nan"),
            ),
        ),
    )
    summary = context_gap_summary(row.bins)

    payload = audit_json_payload(
        rows_by_split={"test": (row,)},
        split_summaries=(),
        model_path=Path("model.pt"),
        model_cache_dir=Path("model-cache"),
        context_cache_dir=Path("context-cache"),
        encoder_sidecar_path=Path("sidecar.npz"),
        prediction_cache=Path("predictions.npy"),
        audit_split="test",
        updated="2026-06-04",
    )

    test_payload = payload["splits"]["test"]
    assert test_payload["rows"][0]["is_flagged"] is True
    assert test_payload["rows"][0]["bins"][0]["bootstrap_samples"] == 32
    assert test_payload["rows"][0]["bins"][0]["gap_ci95_low"] == pytest.approx(0.01)
    assert test_payload["rows"][0]["bins"][1]["empirical_wr"] is None
    assert test_payload["rows"][0]["level_gap"] == pytest.approx(0.10)
    assert test_payload["rows"][0]["slope_gap"] is None
    assert test_payload["rows"][0]["tail_gap"] == pytest.approx(0.10)
    assert test_payload["rows"][0]["tail_bin_label"] == "0"
    assert test_payload["rows"][0]["tail_bin_n"] == 10
    assert test_payload["rows"][0]["direction_correct"] is None
    assert test_payload["flagged_summary"]["support_weighted_gap_mse"] == pytest.approx(
        summary["support_weighted_gap_mse"]
    )
    assert test_payload["flagged_summary"]["n_focus_rows"] == 10


def test_context_examples_support_weighted_gap_math() -> None:
    summary = context_gap_summary(
        (
            AuditBin("a", n=10, empirical_wr=0.5, hgnn_wr=0.6, gap=0.1),
            AuditBin("b", n=30, empirical_wr=0.5, hgnn_wr=0.3, gap=-0.2),
        )
    )

    assert summary["mean_abs_gap"] == pytest.approx(0.15)
    assert summary["support_weighted_mean_abs_gap"] == pytest.approx(0.175)
    assert summary["gap_mse"] == pytest.approx(0.025)
    assert summary["support_weighted_gap_mse"] == pytest.approx(0.0325)


def test_context_examples_audit_slices_full_prediction_cache_to_split(tmp_path) -> None:
    (tmp_path / "cache_meta.json").write_text(
        """
        {
          "n_games": 4,
          "splits": {"train": 1, "test": 3},
          "split_order": ["train", "test"],
          "split_ranges": {
            "train": {"start": 0, "stop": 1},
            "test": {"start": 1, "stop": 4}
          },
          "identity": {"build_vocab": ["ability_power"]}
        }
        """,
        encoding="utf-8",
    )
    np.save(tmp_path / "blue_win.npy", np.array([1.0, 0.0, 1.0, 0.0]))
    np.save(tmp_path / "champion_id.npy", np.zeros((4, 10), dtype=np.int64))
    np.save(tmp_path / "build_id.npy", np.zeros((4, 10), dtype=np.int64))
    np.save(tmp_path / "identity_context_raw.npy", np.zeros((4, 10, 14), dtype=np.float32))
    probabilities = np.arange(40, dtype=np.float32).reshape(4, 10) / 100.0

    data = AuditData(
        context_cache_dir=tmp_path,
        blue_probability=probabilities,
        audit_split="test",
    )

    assert data.n_games == 3
    assert data.blue_win.tolist() == [0.0, 1.0, 0.0]
    assert np.allclose(data.predictions, probabilities[1:4])


def test_context_examples_predict_split_passes_residual_feature_inputs(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_build_hgnn_inputs(**kwargs):
        captured.append(kwargs)
        return {"batch_size": kwargs["champion_id"].shape[0]}

    class FakeModel:
        def __call__(self, **inputs):
            return {"final_logit": torch.zeros(int(inputs["batch_size"]))}

    monkeypatch.setattr(
        "app.ml.context_examples_audit.build_hgnn_inputs",
        fake_build_hgnn_inputs,
    )
    split = SplitData(
        win_rate=np.zeros((3, 10), dtype=np.float32),
        p1_cnt=np.zeros((3, 10), dtype=np.float32),
        blue_win=np.array([1.0, 0.0, 1.0]),
        champion_id=np.zeros((3, 10), dtype=np.int64),
        build_id=np.zeros((3, 10), dtype=np.int64),
        loadout_features=np.arange(30, dtype=np.float32).reshape(3, 10),
        patch_features=np.arange(6, dtype=np.float32).reshape(3, 2),
    )

    probabilities = _predict_split(
        FakeModel(),
        split,
        batch_size=2,
        strength=30.0,
        device="cpu",
    )

    assert probabilities.shape == (3, 10)
    assert np.array_equal(captured[0]["loadout_features"], split.loadout_features[:2])
    assert np.array_equal(captured[0]["patch_features"], split.patch_features[:2])
    assert np.array_equal(captured[1]["loadout_features"], split.loadout_features[2:])
    assert np.array_equal(captured[1]["patch_features"], split.patch_features[2:])


def test_context_examples_audit_focus_probabilities_use_slot_deltas() -> None:
    fallback = _focus_side_probabilities_from_outputs(
        {"final_logit": torch.tensor([0.0, 2.0])}
    )

    assert fallback.shape == (2, 10)
    assert torch.allclose(fallback[0, :5], torch.full((5,), 0.5))
    assert torch.allclose(fallback[0, 5:], torch.full((5,), 0.5))
    assert torch.allclose(fallback[1, :5], torch.full((5,), torch.sigmoid(torch.tensor(2.0))))
    assert torch.allclose(fallback[1, 5:], torch.full((5,), torch.sigmoid(torch.tensor(-2.0))))

    slot_delta = torch.zeros((1, 10))
    slot_delta[0, 0] = 2.0
    slot_delta[0, 5] = -2.0
    focused = _focus_side_probabilities_from_outputs(
        {
            "base_logit": torch.zeros(1),
            "context_logit": torch.zeros(1),
            "semantic_moe_logit": torch.zeros(1),
            "semantic_moe_slot_delta": slot_delta,
        }
    )

    assert focused.shape == (1, 10)
    assert focused[0, 0] > focused[0, 1]
    assert focused[0, 5] < focused[0, 6]

    with_feature_logit = {
        "base_logit": torch.tensor([0.2]),
        "context_logit": torch.tensor([0.1]),
        "semantic_moe_logit": torch.tensor([0.05]),
        "feature_logit": torch.tensor([0.4]),
        "semantic_moe_slot_delta": slot_delta,
    }
    without_feature_logit = {
        key: value for key, value in with_feature_logit.items() if key != "feature_logit"
    }

    audit_focused = _focus_side_probabilities_from_outputs(with_feature_logit)
    train_focused = _train_focus_side_probabilities_from_outputs(with_feature_logit)

    assert torch.allclose(audit_focused, train_focused)
    assert not torch.allclose(
        audit_focused,
        _focus_side_probabilities_from_outputs(without_feature_logit),
    )
