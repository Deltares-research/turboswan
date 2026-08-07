import pathlib
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import tensorflow as tf
import xarray as xr

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint

import create_datasets
import TurboSwanModel

if __name__ == "__main__":
    model_version = "turbo_swan_v0018"

    file_dir = pathlib.Path(__file__).parent
    src_dir = pathlib.Path(__file__).parent.parent.resolve()
    model_dir = src_dir / f"trained_models/{model_version}"
    model_dir.mkdir(exist_ok=True, parents=True)

    shutil.copytree(file_dir, model_dir / file_dir.name, dirs_exist_ok=True)

    tfrecord_dir = pathlib.Path("/gpfs/work2/0/prjs1027/tfrecords/run4")

    train_dir = tfrecord_dir / "train_data"
    val_dir = tfrecord_dir / "val_data"
    test_dir = tfrecord_dir / "test_data"

    output_variables = ["hs"]
    input_variables_original = ["xwnd", "ywnd", "wl"]
    input_variables = ["xwnd", "ywnd", "wl"]
    boundary_variables = ["hs_bnd", "tp_bnd", "theta0_bnd"]
    variable_fields = input_variables_original + output_variables + boundary_variables

    valid_mask = np.load(tfrecord_dir / "valid-mask.npy")

    learning_rate = 1e-4
    n_epochs = 50
    batch_size = 32
    steps_per_execution = 1

    raster_shape = (481, 421, 1 * len(input_variables))
    boundary_shape = (106, len(boundary_variables))
    widths = [16, 32, 64, 128, 256, 256]
    block_depth = 3

    seed = 0

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
