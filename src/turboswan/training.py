import argparse
import copy
import pathlib
import re
import tomllib
from typing import Literal

import keras
import numpy as np
import tensorflow as tf
import xarray as xr
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
    field_validator,
    model_validator,
)

from turboswan.core import TurboSwanModel, create_datasets

# Single source of truth for the raster resolution; threaded into create_datasets/TurboSwanModel.
RASTER_HEIGHT = 481
RASTER_WIDTH = 421
BOUNDARY_POINTS = 106

class _Section(BaseModel):
    # extra="forbid" turns a config typo into an error instead of a silently ignored key.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class DataConfig(_Section):
    # Folder that already contains the data; nothing is downloaded.
    source: str = Field(min_length=1)
    train_subdir: str = "train_data"
    val_subdir: str = "val_data"
    min_max_file: pathlib.Path = pathlib.Path("min_max_files/min-max-values-all.nc")
    valid_mask_file: pathlib.Path = pathlib.Path("valid-mask.npy")
    max_tfrecords: PositiveInt | None = None
    input_variables: list[str] = Field(default=["xwnd", "ywnd", "wl"], min_length=1)
    output_variables: list[str] = Field(default=["hs"], min_length=1)
    boundary_variables: list[str] = ["hs_bnd", "tp_bnd", "theta0_bnd"]
    indices: list[int] = Field(default=[3], min_length=1)
    shuffle_buffer_size: PositiveInt = 1000
    seed: int = 0

    @field_validator("source")
    @classmethod
    def _normalise_source(cls, value: str) -> str:
        return str(pathlib.Path(value).expanduser().resolve())

    @property
    def data_dir(self) -> pathlib.Path:
        """The folder that training reads from."""
        return pathlib.Path(self.source).expanduser().resolve()


class AugmentationConfig(_Section):
    flip_horizontal: bool = False
    flip_vertical: bool = False
    rotate_180: bool = False

    @property
    def enabled(self) -> bool:
        return self.flip_horizontal or self.flip_vertical or self.rotate_180


class ModelConfig(_Section):
    widths: list[PositiveInt] = Field(default=[16, 32, 64, 128, 256, 256], min_length=3)
    block_depth: PositiveInt = 3
    kernel_size: PositiveInt = 5
    dropout: bool = True


class TrainingConfig(_Section):
    learning_rate: PositiveFloat = 1e-4
    lr_decay_steps: PositiveInt = 10000
    lr_decay_rate: PositiveFloat = 0.9
    n_epochs: PositiveInt = 50
    batch_size: PositiveInt = 32
    steps_per_epoch: PositiveInt | None = None
    validation_steps: PositiveInt | None = None
    steps_per_execution: PositiveInt = 1
    verbose: int = 2


class RunConfig(_Section):
    # "local" keeps everything on this machine: data from a folder, MLflow records in a
    # local SQLite folder. "dnai" records to the Data & AI platform's MLflow server.
    mode: Literal["local", "dnai"] = "local"
    model_dir: pathlib.Path
    model_version: str
    tensorboard: bool = True
    histogram_freq: int = Field(default=1, ge=0)
    checkpoint: bool = True
    save_final: bool = True

    @field_validator("model_dir")
    @classmethod
    def _expand(cls, value: pathlib.Path) -> pathlib.Path:
        return value.expanduser().resolve()


class ComparisonConfig(_Section):
    name: str = Field(min_length=1)
    # Folder inside the data source, like data.train_subdir.
    subdir: str = Field(min_length=1)
    plots: list[Literal["Error map"]] = Field(min_length=1)


class MLflowConfig(_Section):
    experiment_name: str
    run_name: str | None = None
    description: str | None = None
    tags: dict[str, str] = {}
    model_name: str = Field(default="model", min_length=1)
    registered_model_name: str | None = None
    # run.mode = "local" only: where the SQLite records go; defaults to ~/turboswan/mlflow.
    local_tracking_dir: pathlib.Path | None = None
    metric_labels: dict[str, str] = Field(default_factory=dict)
    parameter_labels: dict[str, str] = Field(default_factory=dict)
    variable_descriptions: dict[str, str] = Field(default_factory=dict)


class TrainConfig(_Section):
    _recipe_toml: bytes | None = PrivateAttr(default=None)

    data: DataConfig
    run: RunConfig
    augmentation: AugmentationConfig = AugmentationConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    # Read by training_mlflow/training_sweep only; plain training ignores it.
    mlflow: MLflowConfig | None = None
    comparisons: list[ComparisonConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_mode_settings(self) -> "TrainConfig":
        """Reject settings that belong to the other run.mode."""
        if (
            self.run.mode == "dnai"
            and self.mlflow is not None
            and self.mlflow.local_tracking_dir is not None
        ):
            raise ValueError("mlflow.local_tracking_dir only applies when run.mode = 'local'")
        return self


def get_config_path() -> pathlib.Path:
    parser = argparse.ArgumentParser(
        description="Train the TurboSwan model from a TOML configuration file."
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        required=True,
        help="Path to the training config TOML file, e.g. configs/demo.toml.",
    )
    args = parser.parse_args()
    return args.config.resolve()


def load_config_source(config_path: pathlib.Path) -> tuple[dict, bytes]:
    """Read the recipe once, returning its parsed settings and original bytes."""
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found at {config_path}.")

    recipe_toml = config_path.read_bytes()
    return tomllib.loads(recipe_toml.decode("utf-8")), recipe_toml


def load_raw_config(config_path: pathlib.Path) -> dict:
    """Parse a training config TOML file into a raw dict, before pydantic validation."""
    raw, _ = load_config_source(config_path)
    return raw


def load_config(config_path: pathlib.Path) -> TrainConfig:
    raw, recipe_toml = load_config_source(config_path)
    config = TrainConfig.model_validate(raw)
    config._recipe_toml = recipe_toml
    return config


# A TOML table header on a line of its own: [data], [mlflow.metric_labels] or [[comparisons]].
_TABLE_HEADER = re.compile(r"^\s*\[{1,2}\s*([\w.\-]+)\s*\]{1,2}\s*(#.*)?$")


def recipe_section(config: TrainConfig, name: str) -> str:
    """Return one section of the recipe exactly as the researcher wrote it, comments included.

    ``name`` is a section such as ``"data"``. Only that section's own settings are returned;
    its sub-tables (``mlflow.metric_labels``) are left out and named in a closing note, and
    can be asked for by their full name. A repeated ``[[comparisons]]`` table returns every
    entry. A section the recipe leaves out gets a note that its defaults apply.

    Raises:
        ValueError: If the recipe was not loaded from a file, or ``name`` is not a section.
    """
    if config._recipe_toml is None:
        raise ValueError("This configuration was not loaded from a recipe file, so there is no text to show")
    if name.split(".")[0] not in TrainConfig.model_fields:
        raise ValueError(f"Unknown section {name!r}; recipe sections are {sorted(TrainConfig.model_fields)}")

    blocks: list[list[str]] = []
    subtables: list[str] = []
    current: list[str] | None = None
    for line in config._recipe_toml.decode("utf-8").splitlines():
        header = _TABLE_HEADER.match(line)
        if header:
            current = None
            if header.group(1) == name:
                current = [line]
                blocks.append(current)
            elif header.group(1).startswith(f"{name}.") and header.group(1) not in subtables:
                subtables.append(header.group(1))
        elif current is not None:
            current.append(line)

    if not blocks:
        return f"# [{name}] is not set in this recipe, so its defaults apply"
    parts = []
    for block in blocks:
        # Comments just above the next table's header describe that table, not this one.
        while len(block) > 1 and (not block[-1].strip() or block[-1].lstrip().startswith("#")):
            block.pop()
        parts.append("\n".join(block))
    text = "\n\n".join(parts)
    if subtables:
        text += "\n\n# also set in this recipe: " + ", ".join(f"[{table}]" for table in subtables)
    return text


def show_recipe_section(config: TrainConfig, name: str) -> None:
    """Print one section of the recipe as written; see ``recipe_section``."""
    print(recipe_section(config, name))


def apply_overrides(raw: dict, overrides: dict) -> dict:
    """Return a copy of a raw config dict with dotted.key overrides set, e.g. {"training.batch_size": 2}."""
    result = copy.deepcopy(raw)
    for dotted_key, value in overrides.items():
        *sections, key = dotted_key.split(".")
        target = result
        for section in sections:
            target = target[section]
        target[key] = value
    return result


def read_dotted(raw: dict, dotted_key: str, default=None):
    """Inverse of apply_overrides: read raw["training"]["batch_size"] as "training.batch_size"."""
    target = raw
    for part in dotted_key.split("."):
        if not isinstance(target, dict) or part not in target:
            return default
        target = target[part]
    return target


def prepare_output_dir(run_cfg: RunConfig) -> pathlib.Path:
    output_dir = run_cfg.model_dir / run_cfg.model_version
    output_dir.mkdir(exist_ok=True, parents=True)

    return output_dir


def check_inputs(data_cfg: DataConfig) -> pathlib.Path:
    """Return the data folder, raising if it does not exist.

    The data is never downloaded; it must already be in data.source.
    """
    data_dir = data_cfg.data_dir
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data.source {data_dir} does not exist or is not a folder")
    return data_dir


def resolve_split_dir(data_cfg: DataConfig, split: str) -> pathlib.Path:
    subdir = {"train": data_cfg.train_subdir, "val": data_cfg.val_subdir}[split]
    return data_cfg.data_dir / subdir


def resolve_comparison_dir(data_cfg: DataConfig, comparison: ComparisonConfig) -> pathlib.Path:
    return data_cfg.data_dir / comparison.subdir


def load_normalization(data_cfg: DataConfig) -> tuple[xr.Dataset, np.ndarray]:
    """Return the (min/max dataset, valid mask) pair used to normalize and mask samples."""
    min_max_ds = create_datasets.load_min_max_ds(
        data_cfg.data_dir / data_cfg.min_max_file,
        boundary_variables=data_cfg.boundary_variables,
    )
    valid_mask = np.load(data_cfg.data_dir / data_cfg.valid_mask_file)
    return min_max_ds, valid_mask


def list_tfrecords(directory: pathlib.Path, max_tfrecords: int | None) -> list[str]:
    files = sorted(create_datasets.get_files(str(directory)))
    if not files:
        raise FileNotFoundError(f"No .tfrecords files found in {directory}")
    if max_tfrecords is not None:
        files = files[:max_tfrecords]
    return files


def build_dataset(
    directory: pathlib.Path,
    data_cfg: DataConfig,
    min_max_ds: xr.Dataset,
    valid_mask: np.ndarray,
    batch_size: int,
    augmentation_cfg: AugmentationConfig | None = None,
    repeat: bool = False,
) -> tf.data.Dataset:
    """Same pipeline as create_datasets.create_dataset, plus a shard cap and optional repeat()."""
    files = list_tfrecords(directory, data_cfg.max_tfrecords)

    variable_fields = (
        data_cfg.input_variables
        + data_cfg.output_variables
        + data_cfg.boundary_variables
    )

    dataset = create_datasets.get_dataset(
        files,
        min_max_ds=min_max_ds,
        variable_fields=variable_fields,
        input_variables=data_cfg.input_variables,
        output_variables=data_cfg.output_variables,
        boundary_variables=data_cfg.boundary_variables,
        indices=data_cfg.indices,
        raster_height=RASTER_HEIGHT,
        raster_width=RASTER_WIDTH,
        valid_mask=valid_mask,
    )
    dataset = dataset.shuffle(
        buffer_size=data_cfg.shuffle_buffer_size,
        seed=data_cfg.seed,
        reshuffle_each_iteration=True,
    )

    if augmentation_cfg is not None and augmentation_cfg.enabled:
        dataset = dataset.map(
            lambda x, y: create_datasets.augment_pair(
                x,
                y,
                augmentation_cfg.flip_horizontal,
                augmentation_cfg.flip_vertical,
                augmentation_cfg.rotate_180,
            )
        )

    dataset = dataset.batch(batch_size, drop_remainder=True)
    if repeat:
        dataset = dataset.repeat()
    return dataset.prefetch(tf.data.AUTOTUNE)


def get_strategy() -> tf.distribute.Strategy:
    gpus = tf.config.list_logical_devices("GPU")
    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy(
            devices=gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
    return tf.distribute.get_strategy()


def build_model(
    model_cfg: ModelConfig, data_cfg: DataConfig, min_max_ds: xr.Dataset
) -> TurboSwanModel.TurboSwanModel:
    raster_shape = (RASTER_HEIGHT, RASTER_WIDTH, len(data_cfg.input_variables))
    boundary_shape = (BOUNDARY_POINTS, len(data_cfg.boundary_variables))

    return TurboSwanModel.TurboSwanModel(
        input_shape=raster_shape,
        bnd_input_shape=boundary_shape,
        output_variables=data_cfg.output_variables,
        widths=model_cfg.widths,
        block_depth=model_cfg.block_depth,
        k=model_cfg.kernel_size,
        min_max_values=min_max_ds,
        dropout=model_cfg.dropout,
    )


def build_callbacks(run_cfg: RunConfig, output_dir: pathlib.Path) -> list:
    callbacks = []
    model_version = run_cfg.model_version

    if run_cfg.tensorboard:
        callbacks.append(
            keras.callbacks.TensorBoard(
                log_dir=str(output_dir / "logs" / "fit" / model_version),
                histogram_freq=run_cfg.histogram_freq,
            )
        )

    if run_cfg.checkpoint:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(output_dir / f"{model_version}.keras"),
                save_freq="epoch",
                save_best_only=True,
            )
        )

    return callbacks


def run_training(
    config: TrainConfig,
    *,
    extra_callbacks: list[keras.callbacks.Callback] | None = None,
    return_model: bool = False,
) -> keras.callbacks.History | tuple[keras.callbacks.History, keras.Model]:
    if config.comparisons and not return_model:
        raise ValueError(
            "Comparisons require a tracked run; use track_training_run with this recipe"
        )
    data_cfg = config.data
    augmentation_cfg = config.augmentation
    model_cfg = config.model
    training_cfg = config.training
    run_cfg = config.run

    output_dir = prepare_output_dir(run_cfg)

    check_inputs(data_cfg)
    min_max_ds, valid_mask = load_normalization(data_cfg)

    train_dataset = build_dataset(
        resolve_split_dir(data_cfg, "train"),
        data_cfg=data_cfg,
        min_max_ds=min_max_ds,
        valid_mask=valid_mask,
        batch_size=training_cfg.batch_size,
        augmentation_cfg=augmentation_cfg,
        repeat=training_cfg.steps_per_epoch is not None,
    )
    val_dataset = build_dataset(
        resolve_split_dir(data_cfg, "val"),
        data_cfg=data_cfg,
        min_max_ds=min_max_ds,
        valid_mask=valid_mask,
        batch_size=training_cfg.batch_size,
        repeat=training_cfg.validation_steps is not None,
    )

    strategy = get_strategy()
    with strategy.scope():
        model_builder = build_model(model_cfg, data_cfg, min_max_ds)
        model = model_builder.model
        loss = model_builder.masked_loss()

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=training_cfg.learning_rate,
            decay_steps=training_cfg.lr_decay_steps,
            decay_rate=training_cfg.lr_decay_rate,
        )
        opt = keras.optimizers.Adam(learning_rate=lr_schedule)

        model.compile(
            loss=loss,
            optimizer=opt,
            steps_per_execution=training_cfg.steps_per_execution,
        )

        callbacks = build_callbacks(run_cfg, output_dir)
        if extra_callbacks:
            callbacks.extend(extra_callbacks)

        history = model.fit(
            x=train_dataset,
            validation_data=val_dataset,
            epochs=training_cfg.n_epochs,
            steps_per_epoch=training_cfg.steps_per_epoch,
            validation_steps=training_cfg.validation_steps,
            callbacks=callbacks,
            verbose=training_cfg.verbose,
        )

    if run_cfg.save_final:
        model.save(output_dir / f"{run_cfg.model_version}_final.keras")

    if return_model:
        return history, model
    return history


if __name__ == "__main__":
    run_training(load_config(get_config_path()))
