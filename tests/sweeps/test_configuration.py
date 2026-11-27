"""Tests for turboswan.mlops.sweeps.configuration."""

import pytest


def test_pruner_config_requires_optuna_and_a_known_name() -> None:
    pytest.importorskip("optuna")
    from pydantic import ValidationError

    from turboswan.mlops.sweeps.configuration import StrategyConfig

    with pytest.raises(ValidationError):
        StrategyConfig(name="random", max_configs=1, pruner={"name": "MedianPruner"})
    with pytest.raises(ValidationError):
        StrategyConfig(name="optuna", max_configs=1, pruner={"name": "nope"})
    with pytest.raises(ValidationError):
        StrategyConfig(name="optuna", max_configs=1, pruner={"name": "BasePruner"})

    config = StrategyConfig(
        name="optuna",
        max_configs=1,
        pruner={"name": "MedianPruner", "n_startup_trials": 3},
    )
    assert config.pruner.model_extra == {"n_startup_trials": 3}
