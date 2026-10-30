"""Create the human-readable files saved with a tracked run.

These are the plots and tables a researcher opens in the MLflow UI to see what
the model was trained on: the normalization ranges, the water-cell mask, and
the learning curve.
"""

import csv
from collections.abc import Mapping
from pathlib import Path

import keras
import matplotlib.pyplot as plt

from turboswan import training
from turboswan.mlops.tracking.metadata import label_names


def create_training_history_plot(
    history: keras.callbacks.History,
    destination: Path,
    *,
    metric_labels: Mapping[str, str] | None = None,
) -> None:
    """Save a learning-curve image with one line per recorded metric.

    Whatever Keras recorded (loss, validation loss, any extra metric) is drawn,
    so the plot needs no change when the model's metrics do.
    """
    names = label_names(history.history, metric_labels or {})
    figure, axis = plt.subplots(figsize=(8, 4))
    for metric_name, values in history.history.items():
        epochs = range(1, len(values) + 1)
        axis.plot(epochs, values, marker="o", label=names[metric_name])
    axis.set(title="Training history", xlabel="epoch", ylabel="metric value")
    if history.history:
        axis.legend()
    axis.grid(visible=True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def create_preprocessing_artifacts(
    config: training.TrainConfig,
    history: keras.callbacks.History,
    artifact_directory: Path,
    *,
    variable_descriptions: Mapping[str, str] | None = None,
) -> list[Path]:
    """Write the preprocessing tables and plots for a finished run.

    Creates, in ``artifact_directory``:

    * ``normalization-ranges.csv``: each variable's min and max, used to scale
      values to 0-1 before training.
    * ``preprocessing-summary.md``: a short prose summary of normalization and
      the validity mask.
    * ``validity-mask.png``: which grid cells count in the loss.
    * ``learning-curve.png``: metrics per epoch.

    Args:
        config: Validated recipe the run used.
        history: Keras history from training.
        artifact_directory: Folder to write into; created if missing.
        variable_descriptions: Optional meanings for the CSV. They take
            precedence over the recipe's own ``variable_descriptions``.

    Returns:
        Paths to upload to MLflow: the four files above plus the original
        normalization and mask input files.
    """
    descriptions = dict(config.mlflow.variable_descriptions if config.mlflow else {})
    if variable_descriptions is not None:
        descriptions.update(variable_descriptions)
    min_max_ds, valid_mask = training.load_normalization(config.data)
    normalization_path = config.data.data_dir / config.data.min_max_file
    mask_path = config.data.data_dir / config.data.valid_mask_file
    # dict.fromkeys drops duplicates (a variable can be both input and output)
    # while keeping the order the recipe lists them in.
    variables = list(
        dict.fromkeys(
            config.data.input_variables
            + config.data.output_variables
            + config.data.boundary_variables
        )
    )

    artifact_directory.mkdir(parents=True, exist_ok=True)
    # Each variable was scaled to 0-1 as (value - min) / (max - min); the
    # table lets a reader convert model values back to real units.
    ranges_path = artifact_directory / "normalization-ranges.csv"
    with ranges_path.open("w", newline="") as ranges_file:
        writer = csv.DictWriter(
            ranges_file,
            fieldnames=[
                "variable",
                "meaning",
                "minimum",
                "maximum",
                "normalization_note",
            ],
        )
        writer.writeheader()
        for variable in variables:
            writer.writerow(
                {
                    "variable": variable,
                    "meaning": descriptions.get(variable, variable),
                    "minimum": min_max_ds[variable].values[0],
                    "maximum": min_max_ds[variable].values[1],
                    "normalization_note": "(value - minimum) / (maximum - minimum)",
                }
            )

    # The mask has one entry per grid cell: 1 for water, 0 for cells (such as
    # land) that the loss ignores.
    summary_path = artifact_directory / "preprocessing-summary.md"
    valid_cell_count = int(valid_mask.astype(bool).sum())
    total_cell_count = int(valid_mask.size)
    summary_path.write_text(
        f"# TurboSWAN preprocessing inputs\n\n"
        f"## Normalization\n"
        f"Training normalized every selected variable using the ranges in "
        f"`{normalization_path.name}`. See `normalization-ranges.csv` for "
        f"the exact values applied in this run.\n\n"
        f"## Spatial validity mask\n"
        f"`{mask_path.name}` has shape `{valid_mask.shape[0]} x "
        f"{valid_mask.shape[1]}`. It marks {valid_cell_count:,} of "
        f"{total_cell_count:,} grid cells ({valid_cell_count / total_cell_count:.1%}) "
        f"as valid water cells. The masked loss is calculated only over those cells.\n"
    )

    mask_plot_path = artifact_directory / "validity-mask.png"
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(valid_mask, cmap="Blues", vmin=0, vmax=1)
    axis.set(title="Spatial validity mask", xlabel="grid x", ylabel="grid y")
    colorbar = figure.colorbar(image, ax=axis, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["excluded", "used in loss"])
    figure.tight_layout()
    figure.savefig(mask_plot_path, dpi=160)
    plt.close(figure)

    curve_path = artifact_directory / "learning-curve.png"
    create_training_history_plot(
        history,
        curve_path,
        metric_labels=config.mlflow.metric_labels if config.mlflow else None,
    )

    return [
        normalization_path,
        mask_path,
        ranges_path,
        summary_path,
        mask_plot_path,
        curve_path,
    ]