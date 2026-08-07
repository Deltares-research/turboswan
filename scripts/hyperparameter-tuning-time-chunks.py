# %%
import os
import sys

print("Python executable:", sys.executable)
print("Conda environment:", os.environ.get("CONDA_DEFAULT_ENV"))

import pathlib
import random
import shutil

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma

import tensorflow as tf
import xarray as xr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
    Conv2DTranspose,
    Reshape,
    Lambda,
    Cropping2D,
)
from tensorflow.keras.layers import (
    Activation,
    Dropout,
    Dense,
    Flatten,
    Input,
    UpSampling2D,
    Concatenate,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import keras_tuner
# %%
tfrecord_dir = pathlib.Path("/gpfs/work2/0/prjs1027/tfrecords/run3")
output_variables = ["hs", "theta0", "tmm10", "tps", "hswe"]
input_variables = ["xwnd", "ywnd", "wl", "xcur", "ycur"]
boundary_variables = ["hs_bnd", "hswe_bnd", "tmm10_bnd", "theta0_bnd"]
variable_fields = input_variables + output_variables + boundary_variables

min_max_values = xr.open_dataset(tfrecord_dir / "min-max-values/min-max-values-all.nc")
valid_mask = np.load(tfrecord_dir / "valid-mask.npy")
train_dir = tfrecord_dir / 'train_data'
val_dir = tfrecord_dir / 'val_data'
test_dir = tfrecord_dir / 'test_data'

for bnd_var in boundary_variables:
    min_max_values[bnd_var] = min_max_values[bnd_var.split("_")[0]]

# %%
learning_rate = 1e-4
n_epochs = 20
batch_size = 32
steps_per_execution = 1

model_version = "hp-tuning-time-chunks"

file_name = pathlib.Path(__file__).name
src_dir = pathlib.Path(__file__).parent.resolve()
model_dir = src_dir / f"trained_models/{model_version}"
model_dir.mkdir(exist_ok=True, parents=True)
shutil.copy2(pathlib.Path(__file__), model_dir / file_name)

raster_shape = (481, 421, 2 * len(input_variables))
widths = [16, 32, 64, 128, 256, 256]
block_depth = 3


# %%
def get_files(data_path):
    files = tf.io.gfile.glob(data_path + "/" + "*.tfrecords")
    return files


def get_dataset(files):
    """return a tfrecord dataset with all tfrecord files"""
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(tf_parse)
    return dataset


def tf_parse(eg, variable_fields=variable_fields):
    """parse an example (or batch of examples, not quite sure...)"""

    # here we re-specify our format
    # you can also infer the format from the data using tf.train.Example.FromString
    # but that did not work
    parse_dict = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
    }

    vars = variable_fields
    for var in vars:
        parse_dict[var] = tf.io.FixedLenFeature([], tf.string)

    example = tf.io.parse_example(
        eg[tf.newaxis],
        parse_dict,
    )

    variables = {}
    for var in variable_fields:
        variables[var] = tf.io.parse_tensor(example[var][0], out_type="float32")
        variables[var] = (variables[var] - min_max_values[var].values[0]) / (
            min_max_values[var].values[1] - min_max_values[var].values[0]
        )
        variables[var] = tf.where(
            tf.math.is_nan(variables[var]),
            tf.zeros_like(variables[var]) - 9,
            variables[var],
        )
        if var in input_variables:
            variables[var] = variables[var][..., 1:]

        if var in output_variables:
            variables[var] = tf.reshape(variables[var], (481, 421, 1))

    variables["hs_bnd"] = tf.reshape(variables["hs_bnd"], (22, 1))
    variables["tmm10_bnd"] = tf.reshape(variables["tmm10_bnd"], (22, 1))
    variables["hswe_bnd"] = tf.reshape(variables["hswe_bnd"], (22, 1))
    variables["theta0_bnd"] = tf.reshape(variables["theta0_bnd"], (22, 1))

    image_inputs = tf.concat([variables[var] for var in input_variables], axis=-1)
    boundary_inputs = tf.concat(
        [variables["hs_bnd"], variables["tmm10_bnd"], variables["hswe_bnd"], variables["theta0_bnd"]], axis=-1
    )

    inputs = (image_inputs, boundary_inputs)
    outputs = tf.concat([variables[var] for var in output_variables], axis=-1)

    return inputs, outputs


# %%
random.seed(42)
# Get dataset and shuffle
train_files = get_files(str(train_dir))
random.shuffle(train_files)
train_files = train_files[:int(0.5 * len(train_files))]

val_files = get_files(str(val_dir))
test_files = get_files(str(test_dir))

train_dataset = get_dataset(train_files)
val_dataset = get_dataset(val_files)
test_dataset = get_dataset(test_files)

train_dataset = (
    train_dataset.shuffle(buffer_size=1000, seed=0, reshuffle_each_iteration=False)
    .prefetch(batch_size)
    .batch(batch_size, drop_remainder=True)
)

val_dataset = val_dataset.shuffle(buffer_size=1000, seed=0, reshuffle_each_iteration=False).prefetch(batch_size).batch(batch_size, drop_remainder=True)

test_dataset = (
    test_dataset.shuffle(buffer_size=1000, seed=0, reshuffle_each_iteration=False)
    .prefetch(batch_size)
    .batch(batch_size, drop_remainder=True)
)


# %%
def ResidualBlock(width, k, dropout=True):
    def apply(x):
        input_width = x.shape[-1]
        if input_width == width:
            residual = x
        else:
            residual = Conv2D(width, kernel_size=1)(x)

        x = BatchNormalization()(x)
        x = Conv2D(width, kernel_size=k, padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2D(width, kernel_size=k, padding="same")(x)

        x = tf.keras.layers.Add()([x, residual])
        if dropout:
            x = Dropout(0.2)(x)

        return x

    return apply


def ResidualBlockTranspose(width, k, dropout=True):
    def apply(x):
        input_width = x.shape[-1]
        if input_width == width:
            residual = x
        else:
            residual = Conv2D(width, kernel_size=1)(x)

        x = BatchNormalization()(x)
        x = Conv2DTranspose(width, kernel_size=k, padding="same")(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(width, kernel_size=k, padding="same")(x)

        x = tf.keras.layers.Add()([x, residual])
        if dropout:
            x = Dropout(0.2)(x)

        return x

    return apply


def DownBlock(width, k, block_depth, dropout=True):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width, k, dropout)(x)
            skips.append(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        x = tf.keras.layers.Conv2D(
            width, kernel_size=(3, 3), strides=2, padding="same"
        )(x)
        return x

    return apply


def UpBlock(width, k, block_depth, dropout=True):
    def apply(x):
        x, skips = x
        x = tf.keras.layers.Conv2DTranspose(
            width, kernel_size=(3, 3), strides=2, padding="same"
        )(x)
        # x = UpSampling2D(size=2)(x)
        for _ in range(block_depth):
            x = Concatenate()([x, skips.pop()])
            x = ResidualBlockTranspose(width, k, dropout)(x)
        return x

    return apply


# %%
def create_input_model(model_input, widths, k):
    x = ZeroPadding2D(((15, 16), (45, 46)))(model_input)

    x = ResidualBlock(widths[0], k)(x)

    # for width in widths[1: -1]:
    #     x = DownBlock(width, k, block_depth)(x)

    return x


def create_output_model(x, widths, k, name="output"):
    # for width in reversed(widths[1: -1]):
    #     x = UpBlock(width, k, block_depth)(x)

    x = ResidualBlock(widths[0], k)(x)
    model_output = Cropping2D(((15, 16), (45, 46)))(x)

    model_output = Conv2DTranspose(
        len(output_variables), kernel_size=1, padding="valid", activation="linear", name=name
    )(model_output)

    return model_output


# %%
def cnn_model(input_shape, bnd_input_shape, widths, k, block_depth, dropout=True):
    img_input = Input(input_shape)
    x = create_input_model(img_input, widths, k)

    skips = []
    for width in widths[1:-1]:
        x = DownBlock(width, k, block_depth, dropout)([x, skips])
        # for skip_i in range(block_depth):
        #     print(K.int_shape(skips[-skip_i]))

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1], k, dropout)(x)

    img_shape = K.int_shape(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    bnd_input = Input(bnd_input_shape)
    bnd_x = tf.keras.layers.Flatten()(bnd_input)
    bnd_x = tf.keras.layers.Dense(22 * len(boundary_variables))(bnd_x)
    bnd_x = tf.keras.layers.BatchNormalization()(bnd_x)
    bnd_x = tf.keras.layers.Dense(128)(bnd_x)
    bnd_x = tf.keras.layers.BatchNormalization()(bnd_x)
    bnd_x = tf.keras.layers.Dense(256)(bnd_x)
    bnd_x = tf.keras.layers.BatchNormalization()(bnd_x)
    bnd_x = tf.keras.layers.Dense(256)(bnd_x)
    bnd_x = tf.keras.layers.BatchNormalization()(bnd_x)

    x = tf.keras.layers.Concatenate()([x, bnd_x])
    x = tf.keras.layers.Dense(512)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dense(img_shape[1] * img_shape[2] * img_shape[3])(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Reshape((img_shape[1], img_shape[2], img_shape[3]))(x)

    for width in reversed(widths[1:-1]):
        # print(width)
        # for skip_i in range(block_depth):
        #     print(K.int_shape(skips[-skip_i]))
        x = UpBlock(width, k, block_depth, dropout)([x, skips])

    output = create_output_model(x, widths, k)

    # Create model
    model = Model(inputs=[img_input, bnd_input], outputs=[output])

    return model

def denormalize(x, var):
    minimum, maximum = min_max_values[var].values[0], min_max_values[var].values[1]
    x = (x * (maximum - minimum)) + minimum
    return x

def circular_error(a, b, to_rad=True):
    if to_rad:
        a = a * (2 * np.pi) / 360
        b = b * (2 * np.pi) / 360

    a = a % (2 * np.pi)
    b = b % (2 * np.pi)

    diff = tf.math.abs(a - b)    
    return tf.math.minimum(diff, 2 * np.pi - diff) / (2 * np.pi)

def masked_loss(mask):
    def loss(y_true, y_pred):
        mask_expanded = tf.expand_dims(mask, axis=0)
        mask_tiled = tf.tile(mask_expanded, [tf.shape(y_true)[0], 1, 1])

        masked_true = tf.boolean_mask(y_true, mask_tiled)
        masked_pred = tf.boolean_mask(y_pred, mask_tiled)

        theta0_true = denormalize(masked_true[..., 1], var="theta0")
        theta0_pred = denormalize(masked_pred[..., 1], var="theta0")

        diff = masked_true - masked_pred
        diff_theta = circular_error(theta0_true, theta0_pred, to_rad=True)

         # Use TensorFlow operations to modify the tensor
        diff = tf.concat([diff[..., :1], tf.expand_dims(diff_theta, axis=-1), diff[..., 2:]], axis=-1)


        mse = K.mean(tf.square(diff))
        return mse
    return loss

def build_model(hp):
    kernel_size = hp.Int("kernel_size", min_value=3, max_value=5, step=2)
    block_depth = hp.Int("block_depth", min_value=1, max_value=4, step=1)
    learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    model = cnn_model(raster_shape, (22, 4), widths, kernel_size, block_depth)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.9
    )
    opt = Adam(learning_rate=lr_schedule)
    model.compile(
        loss=masked_loss(valid_mask),
        optimizer=opt,
        steps_per_execution=steps_per_execution,
    )
    return model


# %%
# Train model in a session
gpus = tf.config.list_logical_devices("GPU")
strategy = tf.distribute.MirroredStrategy(
    devices=gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
)
with strategy.scope():
    log_dir = str(model_dir / ("logs/fit/" + model_version))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )

    # Define the checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        filepath=str(model_dir / (model_version + '.h5')),  
        save_freq='epoch',
        save_best_only=True                                     
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # prediction_callback = PredictionHistory()
    # debug_callback = DebugCallback()

    tuner = keras_tuner.BayesianOptimization(
        build_model,
        objective="val_loss",
        max_trials=10,
        executions_per_trial=3,
        directory=model_dir,
        project_name="version_1",
    )

    tuner.search(train_dataset, validation_data=val_dataset, epochs=n_epochs, callbacks=[tensorboard_callback, stop_early])

# model.save(model_dir / (model_version + "_final.h5"))
