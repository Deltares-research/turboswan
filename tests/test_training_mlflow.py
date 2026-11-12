"""Tests for the turboswan.training_mlflow entrypoint."""

import math

import pytest


def test_imports_without_cycles() -> None:
    import turboswan.training_mlflow  # noqa: F401


@pytest.mark.e2e
def test_track_training_run_records_everything_on_real_data(e2e_config) -> None:
    from mlflow.models import get_model_info
    from mlflow.tracking import MlflowClient

    from turboswan.mlops.tracking.metadata import default_parameter_labels, recorded_data_source
    from turboswan.training_mlflow import track_training_run

    result = track_training_run(e2e_config)

    client = MlflowClient(result.tracking_uri)
    run = client.get_run(result.run_id)
    n_epochs = e2e_config.training.n_epochs

    assert run.info.status == "FINISHED"
    for metric in ("loss", "val_loss"):
        label = e2e_config.mlflow.metric_labels.get(metric, metric)
        history = client.get_metric_history(result.run_id, label)
        assert [point.step for point in history] == list(range(1, n_epochs + 1))
        assert all(math.isfinite(point.value) for point in history)

    labels = default_parameter_labels(e2e_config)
    assert run.data.params[labels["data.source"]] == recorded_data_source(
        e2e_config, e2e_config.data.data_dir
    )
    assert run.data.params[labels["data.resolved_train_dir"]].endswith(
        e2e_config.data.train_subdir
    )

    artifacts = {item.path for item in client.list_artifacts(result.run_id, "metadata")}
    assert {
        "metadata/config.json",
        "metadata/normalization-ranges.csv",
        "metadata/learning-curve.png",
    } <= artifacts

    assert len(run.inputs.dataset_inputs) == 2
    assert get_model_info(result.model_uri).signature is not None
