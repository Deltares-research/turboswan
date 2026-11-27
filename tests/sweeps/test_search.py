"""Tests for turboswan.mlops.sweeps.search."""

import pytest


def test_random_search_is_deterministic_per_index() -> None:
    from turboswan.mlops.sweeps.configuration import Categorical, Distribution
    from turboswan.mlops.sweeps.search import RandomSearch

    space = {
        "training.batch_size": Categorical(values=[1, 2, 4]),
        "training.learning_rate": Distribution(kind="logfloat", low=1e-5, high=1e-3),
    }
    first = RandomSearch(space, seed=3)
    forward = {k: first.ask(k) for k in range(5)}
    reverse = RandomSearch(space, seed=3)

    assert forward[0] == {}
    for k in (4, 2, 3, 1, 0):
        assert reverse.ask(k) == forward[k]
    assert forward[1] != forward[2]


def test_optuna_report_prunes_until_the_trial_is_told_pruned(optuna_storage: str) -> None:
    optuna = pytest.importorskip("optuna")
    from turboswan.mlops.sweeps.configuration import Categorical, PrunerConfig
    from turboswan.mlops.sweeps.search import OptunaSearch

    search = OptunaSearch(
        {"training.batch_size": Categorical(values=[1, 2])},
        {},
        study_prefix="turboswan",
        pruner=PrunerConfig(name="ThresholdPruner", upper=1.0),
    )
    search.ask(0)

    assert search.report(0, {1: 0.5}) is None
    assert search.report(0, {1: 0.5, 2: 2.0, 3: 3.0}) == 2
    # not closed yet (e.g. the cancel failed): the same prune is reported again
    assert search.report(0, {1: 0.5, 2: 2.0, 3: 3.0, 4: 4.0}) == 2

    study = optuna.load_study(
        study_name=search.mlflow_tags["optuna_study_name"], storage=optuna_storage
    )
    assert study.trials[0].state == optuna.trial.TrialState.RUNNING

    search.tell_pruned(0)

    assert search.report(0, {1: 0.5, 2: 2.0, 3: 3.0, 4: 4.0}) is None
    study = optuna.load_study(
        study_name=search.mlflow_tags["optuna_study_name"], storage=optuna_storage
    )
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    assert study.trials[0].intermediate_values == {1: 0.5, 2: 2.0}


@pytest.mark.usefixtures("optuna_storage")
def test_optuna_search_starts_new_studies_unless_one_is_resumed() -> None:
    pytest.importorskip("optuna")
    from turboswan.mlops.sweeps.configuration import Categorical, Distribution
    from turboswan.mlops.sweeps.search import OptunaSearch

    space = {
        "training.batch_size": Categorical(values=[1, 2, 4]),
        "training.learning_rate": Distribution(kind="logfloat", low=1e-5, high=1e-3),
    }
    base = {
        "training": {"batch_size": 2, "learning_rate": 1e-4},
        "run": {"model_version": "a"},
    }
    # same training/search space keys, different env/runtime keys; 
    # should not change study identity
    moved = {
        "training": {"batch_size": 2, "learning_rate": 1e-4},
        "run": {"model_version": "b", "model_dir": "/elsewhere/models"},
        "data": {"data_dir": "/elsewhere/data"},
        "mlflow": {"experiment_name": "other"},
    }

    first = OptunaSearch(space, base, study_prefix="turboswan")
    study = first.mlflow_tags["optuna_study_name"]
    assert first.index_offset == 0
    assert first.ask(0) == {"training.batch_size": 2, "training.learning_rate": 1e-4}
    first.tell(0, 0.5)
    first.ask(1)
    first.tell(1, 0.2)
    first.ask(2)
    first.tell(2, None)

    second = OptunaSearch(space, base, study_prefix="turboswan")
    assert second.mlflow_tags["optuna_study_name"] != study
    assert second.index_offset == 0

    resumed = OptunaSearch(space, moved, study_prefix="turboswan", resume_study=study)
    assert resumed.mlflow_tags["optuna_study_name"] == study
    assert resumed.index_offset == 3
    assert resumed.prior_best == 0.2
    assert set(resumed.ask(3)) == set(space)


@pytest.mark.usefixtures("optuna_storage")
@pytest.mark.parametrize("name", ["turboswan-000000000000-001", "missing"])
def test_optuna_search_rejects_unknown_or_foreign_studies(name: str) -> None:
    pytest.importorskip("optuna")
    from turboswan.mlops.sweeps.configuration import Categorical
    from turboswan.mlops.sweeps.search import OptunaSearch

    space = {"training.batch_size": Categorical(values=[1, 2])}
    study = OptunaSearch(space, {}, study_prefix="turboswan").mlflow_tags["optuna_study_name"]
    if name == "missing":
        name = study.replace("-001", "-009")

    with pytest.raises(ValueError):
        OptunaSearch(space, {}, study_prefix="turboswan", resume_study=name)


def test_random_strategy_rejects_resume_study() -> None:
    from turboswan.mlops.sweeps.configuration import Categorical, StrategyConfig, SweepConfig
    from turboswan.mlops.sweeps.search import build_search_strategy

    with pytest.raises(ValueError, match="resume_study only applies to the optuna strategy"):
        build_search_strategy(
            SweepConfig(base_config_toml="unused.toml"),
            StrategyConfig(name="random", max_configs=1),
            {"training.batch_size": Categorical(values=[1, 2])},
            {},
            resume_study="turboswan-000000000000-001",
        )
