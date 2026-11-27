"""Tests for turboswan.mlops.sweeps.trial."""

from unittest.mock import Mock

import pytest


def test_sweep_modules_import_without_cycles() -> None:
    import turboswan.mlops.sweeps.blocking.driver  # noqa: F401
    import turboswan.mlops.sweeps.stepped.driver  # noqa: F401


def test_run_trial_tracks_overrides_as_a_child_of_the_parent_run(
    base_config: dict, fake_trial: Mock
) -> None:
    from turboswan.mlops.sweeps.trial import run_trial

    base_config["mlflow"]["registered_model_name"] = "model"
    overrides = {"training.batch_size": 4}

    result = run_trial(
        base_config, overrides, 2, sweep_id="s", parent_run_id="p", recipe_toml=b"recipe"
    )
    config = fake_trial.call_args.args[0]

    assert fake_trial.call_args.kwargs == {"nested": True}
    assert result == {"config_index": 2, "overrides": overrides, "run_id": "run", "val_loss": 0.25}
    assert config.training.batch_size == 4
    assert config.mlflow.registered_model_name is None
    assert config.mlflow.run_name == "s-cfg02"
    assert config.mlflow.tags == {
        "run_type": "sweep_trial",
        "sweep_id": "s",
        "config_index": "2",
        "mlflow.parentRunId": "p",
    }
    assert config._recipe_toml == b"recipe"


def test_run_trial_accepts_the_recipe_toml_as_text(base_config: dict, fake_trial: Mock) -> None:
    from turboswan.mlops.sweeps.trial import run_trial

    run_trial(base_config, {}, 0, sweep_id="s", parent_run_id="p", recipe_toml="# recipé\n")

    assert fake_trial.call_args.args[0]._recipe_toml == "# recipé\n".encode("utf-8")


def test_run_trial_rejects_missing_tracking_config_before_training(
    base_config: dict, fake_trial: Mock
) -> None:
    from turboswan.mlops.sweeps.trial import run_trial

    del base_config["mlflow"]

    with pytest.raises(ValueError, match=r"\[mlflow\]"):
        run_trial(base_config, {}, 0, sweep_id="test", parent_run_id="parent")
    fake_trial.assert_not_called()
