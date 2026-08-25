import argparse
import shutil
import pathlib
import tomllib

import numpy as np

import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

import create_datasets
import TurboSwanModel


def get_config_path() -> pathlib.Path:
    file_dir = pathlib.Path(__file__).parent
    default_config_path = file_dir / "config.toml"

    parser = argparse.ArgumentParser(
        description="Train the TurboSwan model from a TOML configuration file."
    )
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=default_config_path,
        help=(
            "Path to the training config TOML file "
            f"(default: {default_config_path})."
        ),
    )
    args = parser.parse_args()
    return args.config.resolve()


def load_training_config(config_path: pathlib.Path) -> dict:
    if not config_path.exists():
        example_path = config_path.parent / "config.example.toml"
        raise FileNotFoundError(
            "Training config not found at "
            f"{config_path}. Create it from {example_path} or pass --config."
        )

    with config_path.open("rb") as config_file:
        config = tomllib.load(config_file)

    training_config = config.get("training")
    if training_config is None:
        raise KeyError("Missing [training] section in config file.")

    required_keys = [
        "model_dir",
        "model_version",
        "tfrecord_dir",
        "learning_rate",
        "n_epochs",
        "batch_size",
        "seed",
    ]
    missing_keys = [key for key in required_keys if key not in training_config]
    if missing_keys:
        raise KeyError(
            "Missing required training config values: " + ", ".join(missing_keys)
        )

    return training_config


if __name__ == "__main__":
    file_dir = pathlib.Path(__file__).parent
    config_path = get_config_path()
    training_config = load_training_config(config_path)

    model_version = training_config["model_version"]
    model_dir = (
        pathlib.Path(training_config["src_dir"]).expanduser().resolve()
        / f"trained_models/{model_version}"
    )
    model_dir.mkdir(exist_ok=True, parents=True)

    shutil.copytree(file_dir, model_dir / file_dir.name, dirs_exist_ok=True)

    tfrecord_dir = pathlib.Path(training_config["tfrecord_dir"]).expanduser().resolve()

    train_dir = tfrecord_dir / "train_data"
    val_dir = tfrecord_dir / "val_data"
    test_dir = tfrecord_dir / "test_data"

    output_variables = ["hs"]
    input_variables_original = ["xwnd", "ywnd", "wl"]
    input_variables = ["xwnd", "ywnd", "wl"]
    boundary_variables = ["hs_bnd", "tp_bnd", "theta0_bnd"]
    variable_fields = input_variables_original + output_variables + boundary_variables

    valid_mask = np.load(tfrecord_dir / "valid-mask.npy")

    learning_rate = training_config["learning_rate"]
    n_epochs = training_config["n_epochs"]
    batch_size = training_config["batch_size"]
    steps_per_execution = 1

    raster_shape = (481, 421, 1 * len(input_variables))
    boundary_shape = (106, len(boundary_variables))
    widths = [16, 32, 64, 128, 256, 256]
    block_depth = 3

    seed = training_config["seed"]

    min_max_ds = create_datasets.load_min_max_ds(
        tfrecord_dir / "min_max_files/min-max-values-all.nc",
        boundary_variables=boundary_variables,
    )
    train_dataset = create_datasets.create_dataset(
        train_dir,
        min_max_ds=min_max_ds,
        variable_fields=variable_fields,
        input_variables=input_variables,
        output_variables=output_variables,
        boundary_variables=boundary_variables,
        indices=[3],
        valid_mask=valid_mask,
        batch_size=batch_size,
        seed=seed,
        flip_horizontal=False,
        flip_vertical=False,
        rotate_180=False,
    )
    test_dataset = create_datasets.create_dataset(
        val_dir,
        min_max_ds=min_max_ds,
        variable_fields=variable_fields,
        input_variables=input_variables,
        output_variables=output_variables,
        boundary_variables=boundary_variables,
        indices=[3],
        valid_mask=valid_mask,
        batch_size=batch_size,
        seed=seed,
        flip_horizontal=False,
        flip_vertical=False,
        rotate_180=False,
    )

    gpus = tf.config.list_logical_devices("GPU")
    strategy = tf.distribute.MirroredStrategy(
        devices=gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
    with strategy.scope():
        model_builder = TurboSwanModel.TurboSwanModel(
            input_shape=raster_shape,
            bnd_input_shape=boundary_shape,
            output_variables=output_variables,
            widths=widths,
            block_depth=block_depth,
            k=5,
            min_max_values=min_max_ds,
            dropout=True,
        )
        model = model_builder.model
        loss = model_builder.masked_loss()

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.9
        )
        opt = Adam(learning_rate=lr_schedule)

        model.compile(
            loss=loss,
            optimizer=opt,
            steps_per_execution=steps_per_execution,
        )

        log_dir = str(model_dir / ("logs/fit/" + model_version))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )

        # Define the checkpoint callback
        checkpoint_callback = ModelCheckpoint(
            filepath=str(model_dir / (model_version + ".h5")),
            save_freq="epoch",
            save_best_only=True,
        )
        # prediction_callback = PredictionHistory()
        # debug_callback = DebugCallback()

        model.fit(
            x=train_dataset,
            validation_data=test_dataset,
            epochs=n_epochs,
            callbacks=[tensorboard_callback, checkpoint_callback],
            verbose=2,
        )  # , prediction_callback])

    model.save(model_dir / (model_version + "_final.h5"))
