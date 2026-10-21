"""Coordinate one complete TurboSWAN training run recorded in MLflow."""

from collections.abc import Mapping
from dataclasses import dataclass

import keras
import mlflow
import mlflow.keras as mlflow_keras

from turboswan import training
from turboswan.mlops.tracking.comparisons import log_comparisons, validate_comparisons
from turboswan.mlops.tracking.metadata import (
    build_run_metadata,
)
from turboswan.mlops.tracking.mlflow import (
    MetricsCallback,
    build_model_signature,
    configure_run_tracking,
    log_preprocessing_artifacts,
    log_tfrecord_inputs,
)


@dataclass(frozen=True)
class TrackedTrainingResult:
    """What you get back after a tracked run, so you can find it again later.

    Attributes:
        run_id: The MLflow run holding everything recorded for this training.
        tracking_uri: Where that run is stored (the platform, or a local SQLite file).
        model_uri: Address the saved model can be loaded from.
        history: Keras's per-epoch loss/metric values, still in memory.
        model_id: MLflow's id for the logged model, if MLflow provides one.
    """

    run_id: str
    tracking_uri: str
    model_uri: str
    history: keras.callbacks.History
    model_id: str | None = None


def track_training_run(
    config: training.TrainConfig,
    *,
    nested: bool = False,
    variable_descriptions: Mapping[str, str] | None = None,
) -> TrackedTrainingResult:
    """Train TurboSWAN once and record everything about the run in MLflow.

    The training itself is delegated to ``training.run_training``, the same code
    the plain command-line path uses. This function wraps it with the
    record-keeping: it opens an MLflow run, trains inside it, then saves the
    settings, plots, input-file lists and the trained model into that run.

    Args:
        config: Validated recipe. It must contain an ``[mlflow]`` section.
        nested: Record this run inside an already-open MLflow run. Sweeps use
            this so each trial appears under the sweep's parent run.
        variable_descriptions: Optional plain-language meaning of each variable,
            shown in the saved normalization table. Overrides the recipe's own
            ``variable_descriptions`` for names given in both.

    Returns:
        References to the finished run and its model.

    Raises:
        ValueError: If the recipe has no ``[mlflow]`` section.
    """
    mlflow_config = config.mlflow
    if mlflow_config is None:
        raise ValueError("config has no [mlflow] section; add one with at least experiment_name")
    # Check the data first: the comparison folders live inside it.
    training.check_inputs(config.data)
    # Check the comparison folders now: once the run starts, a late failure
    # would leave a half-finished run behind in MLflow.
    validate_comparisons(config)
    metadata = build_run_metadata(
        config,
        experiment_name=mlflow_config.experiment_name,
        run_name=mlflow_config.run_name,
        description=mlflow_config.description,
        tags=mlflow_config.tags,
    )
    tracking_uri = configure_run_tracking(
        metadata.experiment_name,
        mode=config.run.mode,
        local_tracking_dir=mlflow_config.local_tracking_dir,
    )

    # TODO: reuse the sdk's log_run instead of start_run, add an optional nested parameter
    with mlflow.start_run(run_name=metadata.run_name, nested=nested) as run:
        mlflow.set_tags(metadata.tags)
        # "mlflow.note.content" is the special tag MLflow shows as the run's
        # description box in its UI.
        mlflow.set_tag("mlflow.note.content", metadata.description)
        # Parameters are stored as text, so every value is converted to a string.
        mlflow.log_params({key: str(value) for key, value in metadata.params.items()})
        # MetricsCallback copies each epoch's loss/metrics into MLflow as
        # training progresses, so the curves are visible while it runs.
        training_result = training.run_training(
            config,
            extra_callbacks=[MetricsCallback(mlflow_config.metric_labels)],
            return_model=True,
        )
        if not isinstance(training_result, tuple):
            raise RuntimeError("TurboSWAN training did not return the fitted model")
        history, model = training_result
        # Preserve the fitted model before report generation can fail.
        model_info = mlflow_keras.log_model(
            model,
            name=mlflow_config.model_name,
            # The signature records the shapes the model takes in and gives out.
            signature=build_model_signature(model),
            # Packages needed to load the model again elsewhere.
            pip_requirements=["tensorflow>=2.21.0", "keras>=3.15.1"],
            registered_model_name=mlflow_config.registered_model_name,
        )
        log_tfrecord_inputs(config)
        log_comparisons(config, model)
        log_preprocessing_artifacts(
            config,
            history,
            variable_descriptions=variable_descriptions,
        )

    return TrackedTrainingResult(
        run_id=run.info.run_id,
        tracking_uri=tracking_uri,
        model_uri=model_info.model_uri,
        history=history,
        model_id=model_info.model_id,
    )


if __name__ == "__main__":
    track_training_run(training.load_config(training.get_config_path()))