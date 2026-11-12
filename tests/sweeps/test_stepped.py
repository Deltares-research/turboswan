"""Tests for turboswan.mlops.sweeps.stepped."""

import json
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient


# --- step_sweep: resumable loop that submits jobs to an executor --------------------------


class FakeSlurm:
    """Each job finishes on the poll after the one that submitted it, except ``slow`` ones."""

    def __init__(
        self, failing: frozenset[int] = frozenset(), slow: frozenset[int] = frozenset()
    ) -> None:
        self.failing = failing
        self.slow = slow
        self.in_flight: set[int] = set()
        self.max_in_flight = 0
        self.submitted: list[int] = []
        self.cancelled: list[str] = []
        self.polls = 0

    def connect(self) -> None:
        pass

    def submit(self, config_index: int, **kwargs) -> str:
        self.submitted.append(config_index)
        self.in_flight.add(config_index)
        self.max_in_flight = max(self.max_in_flight, len(self.in_flight))
        return f"job-{config_index}"

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)

    def poll(self, jobs: dict[int, str]) -> dict[int, dict]:
        self.polls += 1
        jobs = {i: job for i, job in jobs.items() if i not in self.slow}
        self.in_flight -= set(jobs)
        return {
            i: {"error": "boom"}
            if i in self.failing
            else {"config_index": i, "overrides": {}, "run_id": f"run-{i}", "val_loss": 1.0 + i}
            for i in jobs
        }


class FakeSlurmWithRuns(FakeSlurm):
    """Also opens the child run a real job would, tagged with its parent and config_index."""

    def __init__(self, experiment_name: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.experiment_name = experiment_name
        self.run_ids: dict[int, str] = {}

    def submit(self, config_index: int, **kwargs) -> str:
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        run = MlflowClient().create_run(
            experiment.experiment_id,
            tags={
                "mlflow.parentRunId": kwargs["parent_run_id"],
                "config_index": str(config_index),
            },
        )
        self.run_ids[config_index] = run.info.run_id
        return super().submit(config_index, **kwargs)


@pytest.fixture
def make_step(tmp_path: Path, base_config: dict):
    """Build a ``step()`` callable that advances the sweep one poll at a time."""
    from turboswan.mlops.sweeps.configuration import StrategyConfig, SweepConfig
    from turboswan.mlops.sweeps.stepped.driver import step_sweep

    def make(executor: FakeSlurm, strategy: StrategyConfig | None = None):
        def step() -> bool:
            return step_sweep(
                SweepConfig(base_config_toml="unused.toml"),
                strategy or StrategyConfig(name="random", max_configs=5, max_concurrent=2),
                {},
                base_config,
                sweep_id="test",
                state_path=tmp_path / "sweep.json",
                executor=executor,
            )

        return step

    return make


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "sweep.json"


def test_step_sweep_runs_until_every_config_is_scored(make_step, state_path: Path) -> None:
    executor = FakeSlurm()
    step = make_step(executor)

    outcomes = [step() for _ in range(6)]
    state = json.loads(state_path.read_text())
    parent = MlflowClient(mlflow.get_tracking_uri()).get_run(state["parent_run_id"])

    assert outcomes == [False, False, False, True, True, True]
    assert executor.max_in_flight == 2
    assert executor.submitted == [0, 1, 2, 3, 4]
    assert [result["config_index"] for result in state["results"]] == [0, 1, 2, 3, 4]
    assert parent.info.status == "FINISHED"
    assert parent.data.tags["winner_run_id"] == "run-0"


def test_step_sweep_submits_the_recipe_toml_as_json_safe_text(tmp_path: Path, base_config: dict) -> None:
    from turboswan.mlops.sweeps.configuration import StrategyConfig, SweepConfig
    from turboswan.mlops.sweeps.stepped.driver import step_sweep

    class Recording(FakeSlurm):
        def submit(self, config_index: int, **kwargs) -> str:
            self.kwargs = kwargs
            return super().submit(config_index, **kwargs)

    executor = Recording()
    step_sweep(
        SweepConfig(base_config_toml="unused.toml"),
        StrategyConfig(name="random", max_configs=1),
        {},
        base_config,
        sweep_id="test",
        state_path=tmp_path / "sweep.json",
        executor=executor,
        recipe_toml=b"# recipe\n",
    )

    assert executor.kwargs["recipe_toml"] == "# recipe\n"
    json.dumps(executor.kwargs)


def test_step_sweep_after_finishing_does_nothing(make_step) -> None:
    executor = FakeSlurm()
    step = make_step(executor)
    while not step():
        pass
    submitted, polls = list(executor.submitted), executor.polls

    assert step() is True
    assert (executor.submitted, executor.polls) == (submitted, polls)


def test_step_sweep_records_failed_outcome_and_still_finishes(
    make_step, state_path: Path
) -> None:
    step = make_step(FakeSlurm(failing=frozenset({1})))
    while not step():
        pass
    results = json.loads(state_path.read_text())["results"]

    assert results[1]["error"] == "boom"
    assert [result["config_index"] for result in results] == [0, 1, 2, 3, 4]


def test_step_sweep_fails_the_parent_run_when_every_config_failed(
    make_step, state_path: Path
) -> None:
    step = make_step(FakeSlurm(failing=frozenset(range(5))))
    with pytest.raises(RuntimeError, match="no config in the sweep finished"):
        while not step():
            pass
    parent_run_id = json.loads(state_path.read_text())["parent_run_id"]

    assert MlflowClient(mlflow.get_tracking_uri()).get_run(parent_run_id).info.status == "FAILED"


def test_step_sweep_closes_the_run_a_failed_job_left_open(
    make_step, base_config: dict
) -> None:
    executor = FakeSlurmWithRuns(
        base_config["mlflow"]["experiment_name"], failing=frozenset({1})
    )
    step = make_step(executor)
    while not step():
        pass
    client = MlflowClient(mlflow.get_tracking_uri())

    assert client.get_run(executor.run_ids[1]).info.status == "FAILED"


def test_step_sweep_records_a_failed_submit_and_resumes_past_it(
    make_step, state_path: Path
) -> None:
    class SubmitFails(FakeSlurm):
        def submit(self, config_index: int, **kwargs) -> str:
            if config_index == 1:
                raise ConnectionError("slurm unreachable")
            return super().submit(config_index, **kwargs)

    with pytest.raises(ConnectionError):
        make_step(SubmitFails())()
    state = json.loads(state_path.read_text())

    assert state["next_index"] == 2
    assert list(state["pending"]) == ["0"]
    assert state["results"][0]["config_index"] == 1
    assert "slurm unreachable" in state["results"][0]["error"]

    executor = FakeSlurm()
    step = make_step(executor)
    while not step():
        pass
    results = json.loads(state_path.read_text())["results"]

    assert executor.submitted == [2, 3, 4]
    assert sorted(result["config_index"] for result in results) == [0, 1, 2, 3, 4]


def test_cancel_sweep_cancels_every_pending_job(make_step, state_path: Path) -> None:
    from turboswan.mlops.sweeps.stepped.driver import cancel_sweep

    executor = FakeSlurm()
    assert cancel_sweep(state_path, executor) == 0

    make_step(executor)()

    assert cancel_sweep(state_path, executor) == 2
    assert executor.cancelled == ["job-0", "job-1"]


def test_step_sweep_prunes_cancels_and_skips_pruned_winner(
    make_step,
    state_path: Path,
    base_config: dict,
    optuna_storage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = pytest.importorskip("optuna")
    from turboswan.mlops.sweeps.configuration import StrategyConfig

    base_config["mlflow"]["metric_labels"] = {"val_loss": "Validation masked MSE"}
    executor = FakeSlurmWithRuns(
        base_config["mlflow"]["experiment_name"], slow=frozenset({1})
    )

    def fake_history(client, run_id: str, metric: str) -> dict[int, float]:
        # trials log under the label, so the raw Keras name has no history
        if metric != "Validation masked MSE":
            return {}
        return {1: 5.0} if run_id == executor.run_ids[1] else {}

    monkeypatch.setattr("turboswan.mlops.sweeps.stepped.driver._epoch_history", fake_history)
    step = make_step(
        executor,
        StrategyConfig(
            name="optuna",
            max_configs=3,
            max_concurrent=2,
            pruner={"name": "ThresholdPruner", "upper": 1.0},
        ),
    )
    while not step():
        pass
    state = json.loads(state_path.read_text())
    client = MlflowClient(mlflow.get_tracking_uri())
    parent = client.get_run(state["parent_run_id"])
    pruned = state["results"][1]
    study = optuna.load_study(
        study_name=parent.data.tags["optuna_study_name"], storage=optuna_storage
    )

    assert executor.cancelled == ["job-1"]
    assert pruned["pruned_at_step"] == 1
    assert client.get_run(pruned["run_id"]).info.status == "KILLED"
    assert study.trials[1].state == optuna.trial.TrialState.PRUNED
    assert parent.data.tags["winner_config_index"] == "0"


def test_step_sweep_leaves_the_trial_running_when_cancelling_a_pruned_job_fails(
    make_step,
    state_path: Path,
    base_config: dict,
    optuna_storage: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    optuna = pytest.importorskip("optuna")
    from turboswan.mlops.sweeps.configuration import StrategyConfig

    base_config["mlflow"]["metric_labels"] = {"val_loss": "Validation masked MSE"}

    class CancelFailsOnce(FakeSlurmWithRuns):
        def cancel(self, job_id: str) -> None:
            if not self.cancelled:
                self.cancelled.append(job_id)
                raise ConnectionError("slurm unreachable")
            super().cancel(job_id)

    executor = CancelFailsOnce(base_config["mlflow"]["experiment_name"], slow=frozenset({1}))

    def fake_history(client, run_id: str, metric: str) -> dict[int, float]:
        return {1: 5.0} if run_id == executor.run_ids.get(1) else {}

    monkeypatch.setattr("turboswan.mlops.sweeps.stepped.driver._epoch_history", fake_history)
    step = make_step(
        executor,
        StrategyConfig(
            name="optuna",
            max_configs=3,
            max_concurrent=2,
            pruner={"name": "ThresholdPruner", "upper": 1.0},
        ),
    )
    step()
    with pytest.raises(ConnectionError):
        step()
    state = json.loads(state_path.read_text())
    study_name = MlflowClient(mlflow.get_tracking_uri()).get_run(
        state["parent_run_id"]
    ).data.tags["optuna_study_name"]
    study = optuna.load_study(study_name=study_name, storage=optuna_storage)

    assert "1" in state["pending"]
    assert study.trials[1].state == optuna.trial.TrialState.RUNNING

    while not step():
        pass
    state = json.loads(state_path.read_text())
    study = optuna.load_study(study_name=study_name, storage=optuna_storage)

    assert state["results"][1]["pruned_at_step"] == 1
    assert study.trials[1].state == optuna.trial.TrialState.PRUNED
