"""Tests for turboswan.mlops.sweeps.parent_run."""

from unittest.mock import Mock

import pytest


def test_selection_value_handles_missing_and_nan_metrics() -> None:
    from turboswan.mlops.sweeps.parent_run import selection_value

    result = {"config_index": 0, "overrides": {}, "run_id": "r"}

    assert selection_value(result, "val_loss") is None
    assert selection_value({**result, "val_loss": float("nan")}, "val_loss") is None
    assert selection_value({**result, "val_loss": 0.25}, "val_loss") == 0.25


@pytest.mark.parametrize(
    ("direction", "prior_best", "registered"),
    [
        ("minimize", None, True),
        ("minimize", 0.3, True),
        ("minimize", 0.1, False),
        ("maximize", 0.3, True),
        ("maximize", 0.5, False),
    ],
)
def test_winner_is_registered_only_if_it_beats_the_studys_prior_best(
    monkeypatch: pytest.MonkeyPatch, direction: str, prior_best: float | None, registered: bool
) -> None:
    from turboswan.mlops.sweeps.configuration import SweepConfig
    from turboswan.mlops.sweeps.parent_run import finalize_sweep

    register = Mock(return_value="7")
    monkeypatch.setattr("turboswan.mlops.sweeps.parent_run.register_model_from_run", register)
    client = Mock()
    results = [
        {"config_index": 0, "run_id": "run-0", "val_loss": 0.2},
        {"config_index": 1, "run_id": "run-1", "val_loss": 0.4},
    ]
    sweep = SweepConfig(base_config_toml="unused.toml", direction=direction)

    finalize_sweep(client, "parent", results, sweep, prior_best, "model")
    tags = {call.args[1]: call.args[2] for call in client.set_tag.call_args_list}

    assert register.called is registered
    assert tags.get("winner_model_version") == ("7" if registered else None)
    assert tags.get("winner_beats_prior_best") == (
        None if prior_best is None else str(registered).lower()
    )


def test_sweep_fails_when_no_trial_finished() -> None:
    from turboswan.mlops.sweeps.configuration import SweepConfig
    from turboswan.mlops.sweeps.parent_run import finalize_sweep

    results = [
        {"config_index": 0, "error": "boom"},
        {"config_index": 1, "pruned_at_step": 2},
    ]

    with pytest.raises(RuntimeError, match="no config in the sweep finished"):
        finalize_sweep(Mock(), "parent", results, SweepConfig(base_config_toml="unused.toml"), None, None)
