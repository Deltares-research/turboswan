"""Tests for the turboswan.training_sweep entrypoint."""

import math

import mlflow
import pytest
from mlflow.tracking import MlflowClient


def test_imports_without_cycles() -> None:
    import turboswan.training_sweep  # noqa: F401


def test_main_passes_original_recipe_before_sweep_overrides(tmp_path, monkeypatch) -> None:
    import sys
    from unittest.mock import Mock

    from turboswan import training_sweep

    recipe_toml = (
        b'# Starting recipe\n[data]\nsource = "/tmp/data"\n'
        b'[run]\nmodel_dir = "/tmp/models"\nmodel_version = "baseline"\n'
        b'[mlflow]\nexperiment_name = "turboswan/test"\n'
    )
    (tmp_path / "base.toml").write_bytes(recipe_toml)
    sweep_path = tmp_path / "sweep.toml"
    sweep_path.write_text(
        '[sweep]\nbase_config_toml = "base.toml"\n'
        '[strategy]\nname = "random"\nmax_configs = 1\n'
    )
    run_sweep = Mock()
    monkeypatch.setattr(training_sweep, "run_sweep", run_sweep)
    monkeypatch.setattr(
        sys, "argv",
        ["training_sweep", "--sweep-config", str(sweep_path), "--sweep-id", "test"],
    )

    assert training_sweep.main() == 0

    assert run_sweep.call_args.kwargs["recipe_toml"] == recipe_toml
    assert run_sweep.call_args.args[3]["run"]["model_version"] == "baseline-test"
    assert run_sweep.call_args.kwargs["sweep_id"] == "test"


def test_main_passes_original_recipe_to_stepped_sweep(tmp_path, monkeypatch) -> None:
    from unittest.mock import Mock

    from turboswan import training_sweep

    recipe_toml = (
        b'[data]\nsource = "/tmp/data"\n'
        b'[run]\nmodel_dir = "/tmp/models"\nmodel_version = "baseline"\n'
        b'[mlflow]\nexperiment_name = "turboswan/test"\n'
    )
    (tmp_path / "base.toml").write_bytes(recipe_toml)
    sweep_path = tmp_path / "sweep.toml"
    sweep_path.write_text(
        '[sweep]\nbase_config_toml = "base.toml"\n'
        '[strategy]\nname = "random"\nmax_configs = 1\n'
    )
    step_sweep = Mock(return_value=True)
    monkeypatch.setattr(training_sweep, "step_sweep", step_sweep)

    exit_code = training_sweep.main([
        "--executor", "local-process", "--sweep-config", str(sweep_path),
        "--sweep-id", "test", "--state-path", str(tmp_path / "state.json"),
    ])

    assert exit_code == 0
    assert step_sweep.call_args.kwargs["recipe_toml"] == recipe_toml


@pytest.mark.e2e
def test_sweep_trial_is_a_finished_child_run_with_a_selectable_metric(e2e_config_dict: dict) -> None:
    from turboswan.mlops.sweeps.blocking.driver import run_sweep
    from turboswan.mlops.sweeps.configuration import StrategyConfig, SweepConfig

    e2e_config_dict["training"]["n_epochs"] = 1

    result = run_sweep(
        SweepConfig(base_config_toml="unused.toml"),
        StrategyConfig(max_configs=1),
        {},
        e2e_config_dict,
        sweep_id="e2e",
    )
    trial = MlflowClient(mlflow.get_tracking_uri()).get_run(result["winner"]["run_id"])

    assert trial.info.status == "FINISHED"
    assert trial.data.tags["mlflow.parentRunId"] == result["parent_run_id"]
    assert math.isfinite(result["winner"]["val_loss"])
