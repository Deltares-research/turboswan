"""Shared fixtures for the sweep tests."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def base_config(tmp_path: Path) -> dict:
    """Minimal raw training config with every output under tmp_path."""
    return {
        "data": {"source": str(tmp_path)},
        "run": {"model_dir": str(tmp_path), "model_version": "test"},
        "training": {},
        "mlflow": {
            "experiment_name": "turboswan/sweep-test",
            "local_tracking_dir": str(tmp_path),
        },
    }


@pytest.fixture
def optuna_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point Optuna at a throwaway sqlite database and return its URL."""
    url = f"sqlite:///{(tmp_path / 'studies.db').as_posix()}"
    monkeypatch.setenv("OPTUNA_STORAGE", url)
    return url


@pytest.fixture
def fake_trial(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Stand in for tracked training; a trial's final val_loss is 1 / batch_size."""

    def track(config, *, nested=False):
        history = {"val_loss": [9.0, 1 / config.training.batch_size]}
        return SimpleNamespace(run_id="run", history=SimpleNamespace(history=history))

    mock = Mock(side_effect=track)
    monkeypatch.setattr("turboswan.mlops.sweeps.trial.track_training_run", mock)
    return mock
