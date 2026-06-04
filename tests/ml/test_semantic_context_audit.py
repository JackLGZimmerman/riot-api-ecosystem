from __future__ import annotations

import numpy as np
import pytest
import torch

from app.ml.context_examples_audit import (
    AuditData,
    AuditBin,
    AuditRow,
    AuditSplitSummary,
    AuditSpec,
    _focus_side_probabilities_from_outputs,
    render_audit,
)
from app.ml.semantic_context_audit import (
    ThresholdBin,
    evaluate_threshold_bins,
    gap_summary,
    render_model_alignment_audit,
    side_row_focus_probabilities,
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
                split="val",
                n_games=20,
                n_tests=2,
                n_populated_bins=6,
                mean_abs_gap=0.02,
                max_abs_gap=0.05,
                gap_mse=0.0009,
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
    assert "## Train, Validation, And Test Summary" in markdown
    assert "| Train | 100 | 1,000 | 2 | 7 | 1.00 pp | 4.00 pp | 4.00 pp^2 |" in markdown
    assert "| Validation | 20 | 200 | 2 | 6 | 2.00 pp | 5.00 pp | 9.00 pp^2 |" in markdown
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


def test_context_examples_audit_slices_full_prediction_cache_to_split(tmp_path) -> None:
    (tmp_path / "cache_meta.json").write_text(
        """
        {
          "n_games": 4,
          "splits": {"train": 1, "val": 2, "test": 1},
          "split_order": ["train", "val", "test"],
          "split_ranges": {
            "train": {"start": 0, "stop": 1},
            "val": {"start": 1, "stop": 3},
            "test": {"start": 3, "stop": 4}
          },
          "identity": {"build_vocab": ["ability_power"]}
        }
        """,
        encoding="utf-8",
    )
    np.save(tmp_path / "blue_win.npy", np.array([1.0, 0.0, 1.0, 0.0]))
    np.save(tmp_path / "champion_id.npy", np.zeros((4, 10), dtype=np.int64))
    np.save(tmp_path / "build_id.npy", np.zeros((4, 10), dtype=np.int64))
    np.save(tmp_path / "identity_context_raw.npy", np.zeros((4, 10, 8), dtype=np.float32))
    probabilities = np.arange(40, dtype=np.float32).reshape(4, 10) / 100.0

    data = AuditData(
        context_cache_dir=tmp_path,
        blue_probability=probabilities,
        audit_split="val",
    )

    assert data.n_games == 2
    assert data.blue_win.tolist() == [0.0, 1.0]
    assert np.allclose(data.predictions, probabilities[1:3])


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
