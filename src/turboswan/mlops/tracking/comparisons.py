"""Add prediction-versus-SWAN comparison plots to a tracked run.

A "comparison" is a recipe entry pointing at a folder of examples. After
training, the model predicts one example from each folder and the result is
plotted against the SWAN simulation. This is only for inspection: it does not
affect training.
"""

from pathlib import Path

import keras
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from turboswan import training
from turboswan.core import create_datasets
from turboswan.mlops.tracking.metadata import label_names


def validate_comparisons(config: training.TrainConfig) -> None:
    """Check the comparison folders exist and have data, before training starts.

    Run first so that a typo in a comparison path is reported immediately,
    instead of after hours of training.

    Raises:
        ValueError: If comparisons are set but ``hs`` is not an output variable.
        FileNotFoundError: If a comparison folder is missing or holds no shards.
    """
    if not config.comparisons:
        return
    # The error map is of wave height (hs), so the model must predict it.
    if "hs" not in config.data.output_variables:
        raise ValueError("Error map requires 'hs' in data.output_variables")
    for comparison in config.comparisons:
        directory = training.resolve_comparison_dir(config.data, comparison)
        if not directory.is_dir():
            raise FileNotFoundError(
                f"Comparison '{comparison.name}' directory not found: {directory}"
            )
        training.list_tfrecords(directory, None)


def log_comparisons(config: training.TrainConfig, model: keras.Model) -> None:
    """Plot SWAN minus TurboSWAN wave height for one example per comparison.

    For each comparison folder this takes the first example in the first shard,
    lets the trained model predict it, and logs a map of the difference as an
    image in MLflow (purple where SWAN is higher, orange where TurboSWAN is
    higher), plus a few parameters saying which folder and file were used.
    """
    if not config.comparisons:
        return
    validate_comparisons(config)
    min_max_ds, valid_mask = training.load_normalization(config.data)
    data_cfg = config.data
    for index, comparison in enumerate(config.comparisons, start=1):
        files = training.list_tfrecords(
            training.resolve_comparison_dir(data_cfg, comparison), None
        )
        # Build a dataset from just the first shard, then take one example.
        dataset = create_datasets.get_dataset(
            files[:1],
            min_max_ds=min_max_ds,
            variable_fields=(
                data_cfg.input_variables
                + data_cfg.output_variables
                + data_cfg.boundary_variables
            ),
            input_variables=data_cfg.input_variables,
            output_variables=data_cfg.output_variables,
            boundary_variables=data_cfg.boundary_variables,
            indices=data_cfg.indices,
            raster_height=training.RASTER_HEIGHT,
            raster_width=training.RASTER_WIDTH,
            valid_mask=valid_mask,
        ).batch(1)
        example = next(iter(dataset.take(1)), None)
        if example is None:
            raise ValueError(
                f"Comparison '{comparison.name}' has no examples in {files[0]}"
            )
        inputs, targets = example
        predictions = model.predict(inputs, verbose=0)

        # Both the targets and the predictions are normalized to 0-1 during
        # training. Reverse that with the variable's recorded min and max so
        # the map is in metres.
        hs_index = data_cfg.output_variables.index("hs")
        minimum, maximum = min_max_ds["hs"].values
        target_values = targets.numpy()
        swan = target_values[0, ..., hs_index] * (maximum - minimum) + minimum
        turboswan = predictions[0, ..., hs_index] * (maximum - minimum) + minimum
        # The dataset appends the validity mask as the last target channel;
        # zero marks cells that are not water. Also skip any NaN/infinite cell.
        mask = target_values[0, ..., -1] != 0
        valid = mask & np.isfinite(swan) & np.isfinite(turboswan)
        if not valid.any():
            raise ValueError(
                f"Comparison '{comparison.name}' has no valid wave-height cells"
            )
        difference = np.where(valid, swan - turboswan, np.nan)

        params = {
            "name": comparison.name,
            "subdir": comparison.subdir,
            "sample_file": Path(files[0]).name,
            "reference": "SWAN hs TFRecord target",
        }
        # Same rule as for other settings: if the recipe's parameter_labels
        # lists comparison fields, only those are recorded, under those labels.
        # A label may contain "{index}" to tell several comparisons apart.
        labels = config.mlflow.parameter_labels if config.mlflow else {}
        if labels:
            params = {
                field: value for field, value in params.items()
                if f"comparison.{field}" in labels
            }
        comparison_params = {
            f"comparison.{index}.{field}": value for field, value in params.items()
        }
        names = label_names(
            comparison_params,
            {
                f"comparison.{index}.{field}": labels[f"comparison.{field}"].format(index=index)
                for field in params if f"comparison.{field}" in labels
            },
        )
        mlflow.log_params({names[name]: value for name, value in comparison_params.items()})
        # Centre the colour scale on zero and make it symmetric, so "no
        # difference" is always the neutral colour. The tiny floor avoids a
        # zero-width scale if the two outputs happen to match exactly.
        magnitude = max(float(np.nanmax(np.abs(difference))), np.finfo(float).eps)
        figure, axis = plt.subplots(figsize=(8, 6))
        # Close the figure even on error, or matplotlib keeps it in memory.
        try:
            image = axis.imshow(
                difference, cmap="PuOr", vmin=-magnitude, vmax=magnitude
            )
            axis.set(
                title=f"{comparison.name}: SWAN minus TurboSWAN (hs)",
                xlabel="grid x",
                ylabel="grid y",
            )
            figure.colorbar(image, ax=axis, label="wave-height difference (m)")
            figure.tight_layout()
            mlflow.log_figure(figure, f"comparisons/{index:02d}/error-map-hs.png")
        finally:
            plt.close(figure)
