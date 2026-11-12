"""End-to-end tests for turboswan.training on local demo data."""

import math

import pytest

pytestmark = pytest.mark.e2e


def test_run_training_fits_on_real_data(e2e_config) -> None:
    from keras.callbacks import History

    from turboswan import training

    history = training.run_training(e2e_config)
    assert isinstance(history, History)

    n_epochs = e2e_config.training.n_epochs
    for metric in ("loss", "val_loss"):
        assert len(history.history[metric]) == n_epochs
        assert all(math.isfinite(value) for value in history.history[metric])


def test_run_training_returns_the_fitted_model_on_request(e2e_config) -> None:
    from turboswan import training

    result = training.run_training(e2e_config, return_model=True)
    assert isinstance(result, tuple)
    history, model = result

    assert history.history["loss"]
    assert model.inputs


def test_run_training_saves_the_final_model(e2e_config) -> None:
    from turboswan import training

    e2e_config.run.save_final = True

    training.run_training(e2e_config)

    final_model = (
        e2e_config.run.model_dir
        / e2e_config.run.model_version
        / f"{e2e_config.run.model_version}_final.keras"
    )
    assert final_model.is_file()
