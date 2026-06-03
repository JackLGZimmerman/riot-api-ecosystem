from __future__ import annotations

import numpy as np
import pytest

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
