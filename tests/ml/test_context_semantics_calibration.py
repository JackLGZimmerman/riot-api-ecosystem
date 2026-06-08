from pathlib import Path

from app.ml.experiments.context_semantics_calibration import (
    SPEC_BY_NAME,
    selected_specs,
    spec_payload,
    train_command,
)


def test_relaxed_context_residual_spec_passes_teacher_amplitude_flags(
    tmp_path: Path,
) -> None:
    spec = SPEC_BY_NAME["group_context_tail_uncert_huber_relaxed_context"]

    command = train_command(spec, seed=4, experiment_root=tmp_path)

    assert "--semantic-context-calibration-residual-loss" in command
    assert (
        command[command.index("--semantic-context-calibration-residual-loss") + 1]
        == "uncertainty_huber"
    )
    assert (
        command[
            command.index(
                "--semantic-context-calibration-context-residual-shrink-strength"
            )
            + 1
        ]
        == "15000.0"
    )
    assert (
        command[
            command.index("--semantic-context-calibration-context-residual-clip") + 1
        ]
        == "0.08"
    )


def test_context_semantics_spec_payload_exposes_residual_controls() -> None:
    payload = spec_payload(SPEC_BY_NAME["group_context_tail_uncert_huber"])
    relaxed_payload = spec_payload(
        SPEC_BY_NAME["group_context_tail_uncert_huber_relaxed_context"]
    )

    assert payload["context_residual_shrink_strength"] is None
    assert payload["context_residual_clip"] is None
    assert relaxed_payload["context_residual_shrink_strength"] == 15_000.0
    assert relaxed_payload["context_residual_clip"] == 0.08


def test_selected_specs_includes_relaxed_context_residual_candidate() -> None:
    selected = selected_specs(["group_context_tail_uncert_huber_relaxed_context"])

    assert [spec.name for spec in selected] == [
        "group_context_tail_uncert_huber_relaxed_context"
    ]
