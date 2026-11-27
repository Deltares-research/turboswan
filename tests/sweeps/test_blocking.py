"""Tests for turboswan.mlops.sweeps.blocking."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import mlflow
import pytest
from mlflow.tracking import MlflowClient


class FixedSearch:
    """Search stub that proposes batch sizes 1..4 and records the scores it is told."""

    index_offset = 0
    prior_best = None
    mlflow_tags: dict[str, str] = {}

    def __init__(self) -> None:
        self.scores: list[tuple[int, float | None]] = []

    def ask(self, config_index: int) -> dict:
        return {"training.batch_size": [1, 2, 3, 4][config_index]}

    def tell(self, config_index: int, value: float | None) -> None:
        self.scores.append((config_index, value))


def test_sweep_links_ranked_trials_to_one_parent_run(
    tmp_path: Path,
    base_config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from turboswan.mlops.sweeps.blocking.driver import run_sweep
    from turboswan.mlops.sweeps.configuration import StrategyConfig, SweepConfig

    val_losses = {1: 0.5, 2: 0.2, 4: 0.9}

    def fake_track_training_run(config, *, nested=False):
        assert nested is True
        with mlflow.start_run(run_name=config.mlflow.run_name, nested=nested) as run:
            mlflow.set_tags(config.mlflow.tags)
        if config.training.batch_size not in val_losses:
            raise RuntimeError("trial failed")
        history = {"val_loss": [1.0, val_losses[config.training.batch_size]]}
        return SimpleNamespace(
            run_id=run.info.run_id, history=SimpleNamespace(history=history)
        )

    monkeypatch.setattr(
        "turboswan.mlops.sweeps.trial.track_training_run",
        fake_track_training_run,
    )
    search = FixedSearch()
    monkeypatch.setattr(
        "turboswan.mlops.sweeps.blocking.driver.build_search_strategy",
        lambda *args, **kwargs: search,
    )

    result = run_sweep(
        SweepConfig(base_config_toml="unused.toml"),
        StrategyConfig(max_configs=4),
        {},
        base_config,
        sweep_id="test",
    )
    client = MlflowClient(mlflow.get_tracking_uri())
    parent = client.get_run(result["parent_run_id"])
    children = client.search_runs(
        [parent.info.experiment_id],
        filter_string=f"tags.`mlflow.parentRunId` = '{parent.info.run_id}'",
    )

    assert result["winner"]["config_index"] == 1
    assert result["results"][2]["error"] == "RuntimeError('trial failed')"
    assert search.scores == [(0, 0.5), (1, 0.2), (2, None), (3, 0.9)]
    assert len(children) == 4
    assert parent.data.tags["run_type"] == "sweep"
    assert parent.data.tags["winner_run_id"] == result["winner"]["run_id"]
    assert parent.data.metrics["best_val_loss"] == 0.2


def test_sweep_cancels_unfinished_trials_when_interrupted(
    base_config: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from turboswan.mlops.sweeps.blocking.driver import run_sweep
    from turboswan.mlops.sweeps.blocking.executors import SequentialExecutor
    from turboswan.mlops.sweeps.configuration import StrategyConfig, SweepConfig

    executor = SequentialExecutor()
    executor.wait_next = Mock(side_effect=KeyboardInterrupt)
    executor.cancel_all = Mock()
    monkeypatch.setattr(
        "turboswan.mlops.sweeps.blocking.driver.build_search_strategy",
        lambda *args, **kwargs: FixedSearch(),
    )

    with pytest.raises(KeyboardInterrupt):
        run_sweep(
            SweepConfig(base_config_toml="unused.toml"),
            StrategyConfig(max_configs=2),
            {},
            base_config,
            sweep_id="test",
            executor=executor,
        )

    executor.cancel_all.assert_called_once()


@pytest.mark.usefixtures("optuna_storage", "fake_trial")
def test_later_sweeps_start_a_new_study_unless_one_is_resumed(base_config: dict) -> None:
    pytest.importorskip("optuna")
    from turboswan.mlops.sweeps.blocking.driver import run_sweep
    from turboswan.mlops.sweeps.configuration import Categorical, StrategyConfig, SweepConfig

    def sweep(sweep_id: str, **kwargs) -> dict:
        return run_sweep(
            SweepConfig(base_config_toml="unused.toml"),
            StrategyConfig(name="optuna", max_configs=2),
            {"training.batch_size": Categorical(values=[1, 2, 4])},
            base_config,
            sweep_id=sweep_id,
            **kwargs,
        )

    first = sweep("first")
    fresh = sweep("fresh")
    client = MlflowClient(mlflow.get_tracking_uri())
    study = client.get_run(first["parent_run_id"]).data.tags["optuna_study_name"]
    resumed = sweep("resumed", resume_study=study)
    resumed_tags = client.get_run(resumed["parent_run_id"]).data.tags

    assert [result["config_index"] for result in first["results"]] == [0, 1]
    assert [result["config_index"] for result in fresh["results"]] == [0, 1]
    assert [result["config_index"] for result in resumed["results"]] == [2, 3]
    assert resumed_tags["optuna_study_name"] == study
    assert resumed_tags["study_best_before"] == str(first["winner"]["val_loss"])
