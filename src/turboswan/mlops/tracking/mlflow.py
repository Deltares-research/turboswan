"""Track a configured TurboSWAN training run in MLflow.

This module contains the MLflow and Keras operations used by the tracked-run and
sweep workflows. It does not alter model architecture, data meaning, or
validation criteria.
"""

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import keras
import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
from deltares_ai_platform_sdk.mlflow_utils import configure_tracking
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models.signature import ModelSignature
from mlflow.tracking import MlflowClient
from mlflow.types.schema import Schema, TensorSpec

from turboswan import training
from turboswan.mlops.tracking.artifacts import create_preprocessing_artifacts
from turboswan.mlops.tracking.metadata import (
    label_names,
    portable_path,
    recorded_config,
    recorded_data_source,
    tfrecord_manifest_digest,
    validate_identifiers,
)


# TODO: add functionality of configure_local_tracking/configure_run_tracking to sdk's configure_tracking
def configure_local_tracking(experiment_name: str, tracking_dir: Path) -> str:
    """Create or select a local SQLite-backed MLflow experiment.

    Args:
        experiment_name: Name used to group comparable local runs.
        tracking_dir: Directory that will contain ``mlflow.db`` and the
            experiment's ``artifacts`` directory.

    Returns:
        SQLite MLflow tracking URI for the selected experiment.
    """
    tracking_dir = tracking_dir.expanduser().resolve()
    tracking_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = tracking_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    # Run settings, metrics and tags live in one SQLite file; larger files
    # (plots, the model) are stored as ordinary files in ``artifacts/``.
    tracking_uri = f"sqlite:///{tracking_dir / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri)
    # The artifact location can only be chosen when an experiment is first
    # created, so look it up first rather than creating it every time.
    if client.get_experiment_by_name(experiment_name) is None:
        client.create_experiment(
            experiment_name, artifact_location=artifact_dir.as_uri()
        )
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def configure_run_tracking(
    experiment_name: str,
    *,
    mode: Literal["dnai", "local"] = "dnai",
    local_tracking_dir: Path | None = None,
) -> str:
    """Select Data & AI platform or local MLflow tracking for one training run.

    ``mode`` is the recipe's ``run.mode``. ``"dnai"`` uses the platform tracking
    URI and injected credentials. ``"local"`` uses a SQLite database in
    ``local_tracking_dir``, defaulting to ``~/turboswan/mlflow``.
    """
    if mode == "dnai":
        return configure_tracking(experiment_name)
    if mode == "local":
        return configure_local_tracking(
            experiment_name,
            local_tracking_dir or Path.home() / "turboswan" / "mlflow",
        )
    raise ValueError("mode must be either 'dnai' or 'local'")


class MetricsCallback(keras.callbacks.Callback):
    """Send each epoch's loss and metrics to MLflow as training runs.

    Keras calls ``on_epoch_end`` after every epoch with that epoch's numbers
    (for example ``loss`` and ``val_loss``). Logging them with the epoch number
    as the step is what produces the learning curves in the MLflow UI.

    Args:
        metric_labels: Optional recipe mapping from Keras metric names to the
            names shown in MLflow. Unlisted metrics keep their Keras name.
    """

    def __init__(self, metric_labels: Mapping[str, str] | None = None) -> None:
        super().__init__()
        self.metric_labels = dict(metric_labels or {})

    def on_epoch_end(self, epoch: int, logs: Mapping[str, float] | None = None) -> None:
        if logs:
            names = label_names(logs, self.metric_labels)
            metrics = {names[name]: float(value) for name, value in logs.items()}
            validate_identifiers(metric_names=metrics)
            # Keras counts epochs from 0; MLflow steps read better from 1.
            mlflow.log_metrics(metrics, step=epoch + 1)


def build_model_signature(model: keras.Model) -> ModelSignature:
    """Describe the shapes and data types the model takes in and gives out.

    MLflow stores this "signature" with the saved model, so anyone loading it
    later can see what input to supply without reading the training code.
    """

    def tensor_spec(tensor: tf.Tensor, name: str) -> TensorSpec:
        # Keras marks a variable-size dimension (usually the batch size) as
        # None; MLflow's convention for the same thing is -1.
        shape = tuple(
            -1 if dimension is None else int(dimension)
            for dimension in tensor.shape
        )
        return TensorSpec(np.dtype(str(tensor.dtype)), shape, name=name)

    inputs = Schema([tensor_spec(tensor, tensor.name) for tensor in model.inputs])
    outputs = Schema(
        [
            tensor_spec(tensor, name)
            for tensor, name in zip(model.outputs, model.output_names, strict=True)
        ]
    )
    return ModelSignature(inputs=inputs, outputs=outputs)


def log_tfrecord_inputs(config: training.TrainConfig) -> None:
    """Record which data files a run used, as MLflow dataset inputs.

    MLflow does not store the data itself. Dataset inputs hold the folder,
    schema and digest; MLflow tables preserve the selected shard names and paths.

    Args:
        config: Validated training configuration used to resolve the training and
            validation split directories and shard cap.

    Side Effects:
        Logs one MLflow dataset input and a ``metadata/<split>-tfrecords.json``
        table for each split in the current active run.

    Raises:
        FileNotFoundError: If a configured split has no TFRecord shards.
    """
    for split, context in (("train", "training"), ("val", "validation")):
        location = training.resolve_split_dir(config.data, split)
        # Same file selection (including the max_tfrecords cap) as training
        # used, so the recorded list matches what the model actually saw.
        tfrecord_files = training.list_tfrecords(
            location, config.data.max_tfrecords
        )
        manifest = pd.DataFrame(
            {
                "tfrecord_file": [Path(file).name for file in tfrecord_files],
                "file_path": [portable_path(file) for file in tfrecord_files],
            }
        )
        dataset = from_pandas(
            manifest,
            source=recorded_data_source(config, location),
            name=f"TurboSWAN {context} TFRecords",
            digest=tfrecord_manifest_digest(tfrecord_files),
        )
        mlflow.log_table(manifest, f"metadata/{split}-tfrecords.json")
        mlflow.log_input(dataset, context=context)


def log_preprocessing_artifacts(
    config: training.TrainConfig,
    history: keras.callbacks.History,
    *,
    variable_descriptions: Mapping[str, str] | None = None,
) -> None:
    """Write and log configuration, preprocessing, and learning-curve artifacts.

    Args:
        config: Validated configuration used for the completed training run.
        history: Keras history returned by training. Every recorded metric is
            included in the learning-curve artifact.
        variable_descriptions: Researcher-facing variable descriptions used in
            the normalization-ranges CSV.

    Side Effects:
        Logs the effective configuration as ``metadata/config.json`` and the
        original ``metadata/recipe.toml`` when loaded from a recipe. Creates
        temporary CSV, Markdown, and PNG files that are logged, together with the
        normalization inputs, under MLflow's ``metadata`` artifact path.
        Temporary files are removed afterwards.

    Raises:
        FileNotFoundError: If the configured normalization or mask files are
            unavailable.
        KeyError: If a configured variable is absent from normalization data.
    """
    mlflow.log_dict(recorded_config(config), "metadata/config.json")
    # Files are first written to a throwaway folder, uploaded to MLflow, and the
    # folder is deleted when the ``with`` block ends.
    with tempfile.TemporaryDirectory() as temporary_directory:
        # ``_recipe_toml`` holds the recipe file's original bytes, so the exact
        # text the researcher wrote is kept alongside the processed config.
        if config._recipe_toml is not None:
            recipe_path = Path(temporary_directory) / "recipe.toml"
            recipe_path.write_bytes(config._recipe_toml)
            mlflow.log_artifact(str(recipe_path), artifact_path="metadata")
        metadata_files = create_preprocessing_artifacts(
            config,
            history,
            Path(temporary_directory),
            variable_descriptions=variable_descriptions,
        )
        for metadata_file in metadata_files:
            mlflow.log_artifact(str(metadata_file), artifact_path="metadata")
