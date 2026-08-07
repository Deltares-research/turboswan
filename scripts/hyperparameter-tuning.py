# %%
import ast
import pathlib
import random
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

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
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# %%
tfrecord_dir = pathlib.Path("/gpfs/work2/0/prjs1027/tfrecords/run2")
output_variables = ["hs", "theta0_x", "theta0_y", "tmm10", "tps", "hswe"]
input_variables = ["xwnd", "ywnd", "ssh", "xcur", "ycur"]
boundary_variables = ["hs_bnd", "hswe_bnd", "tmm10_bnd"]
variable_fields = input_variables + output_variables + boundary_variables

min_max_values = xr.open_dataset(tfrecord_dir / "min-max-values/min-max-values.nc")

for bnd_var in boundary_variables:
    min_max_values[bnd_var] = min_max_values[bnd_var.split("_")[0]]

min_max_values["theta0_x"] = xr.DataArray(np.array([-1.0, 1.0]), dims="index")
min_max_values["theta0_y"] = xr.DataArray(np.array([-1.0, 1.0]), dims="index")

# %%
n_epochs = 10
steps_per_execution = 1

batch_size = 32
kernel_size = int(sys.argv[1])
learning_rate = float(sys.argv[2])
block_depth = int(sys.argv[3])
print(sys.argv[4])
widths = ast.literal_eval(sys.argv[4])

model_version = "swan_ens_nn_tests-2"

file_name = pathlib.Path(__file__).name
src_dir = pathlib.Path(__file__).parent.resolve()
model_dir = src_dir / f"trained_models/{model_version}"
model_dir.mkdir(exist_ok=True, parents=True)
shutil.copy2(pathlib.Path(__file__), model_dir / file_name)

raster_shape = (481, 421, 2 * len(input_variables))


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
        if var in output_variables:
            variables[var] = variables[var][..., -1]
            variables[var] = tf.reshape(variables[var], (481, 421, 1))

    variables["hs_bnd"] = tf.reshape(variables["hs_bnd"], (22, 1))
    variables["tmm10_bnd"] = tf.reshape(variables["tmm10_bnd"], (22, 1))
    variables["hswe_bnd"] = tf.reshape(variables["hswe_bnd"], (22, 1))

    image_inputs = tf.concat([variables[var] for var in input_variables], axis=-1)
    boundary_inputs = tf.concat(
        [variables["hs_bnd"], variables["tmm10_bnd"], variables["hswe_bnd"]], axis=-1
    )

    inputs = (image_inputs, boundary_inputs)
    outputs = tf.concat([variables[var] for var in output_variables], axis=-1)

    return inputs, outputs


# %%
# Get dataset and shuffle
random.seed(0)
files = get_files(str(tfrecord_dir))
random.shuffle(files)
train_files = files[:int(0.7*len(files))]
train_files = train_files[:100]

test_files = files[int(0.7 * len(files)): ]
test_files = files[:30]
dataset = get_dataset(train_files)
val_dataset = get_dataset(test_files)
# Filter out all samples with high values
# dataset = dataset.filter(lambda x, y: tf.reduce_all(x <= 1))
# dataset = dataset.filter(lambda x, y: tf.reduce_all(y <= 10))

train_dataset = (
    dataset.shuffle(buffer_size=1000, seed=0, reshuffle_each_iteration=False)
    .prefetch(batch_size)
    .batch(batch_size, drop_remainder=True)
)
test_dataset = (
    val_dataset.shuffle(buffer_size=1000, seed=0, reshuffle_each_iteration=False)
    .prefetch(batch_size)
    .batch(batch_size, drop_remainder=True)
)


# %%
def ResidualBlock(width, k):
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

        return x

    return apply


def ResidualBlockTranspose(width, k):
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

        return x

    return apply


def DownBlock(width, k, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width, k)(x)
            skips.append(x)
        # x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
        x = tf.keras.layers.Conv2D(
            width, kernel_size=(3, 3), strides=2, padding="same"
        )(x)
        return x

    return apply


def UpBlock(width, k, block_depth):
    def apply(x):
        x, skips = x
        x = tf.keras.layers.Conv2DTranspose(
            width, kernel_size=(3, 3), strides=2, padding="same"
        )(x)
        # x = UpSampling2D(size=2)(x)
        for _ in range(block_depth):
            x = Concatenate()([x, skips.pop()])
            x = ResidualBlockTranspose(width, k)(x)
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
def cnn_model(input_shape, bnd_input_shape, widths, k, block_depth):
    img_input = Input(input_shape)
    x = create_input_model(img_input, widths, k)

    skips = []
    for width in widths[1:-1]:
        x = DownBlock(width, k, block_depth)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1], k)(x)

    img_shape = K.int_shape(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    bnd_input = Input(bnd_input_shape)
    bnd_x = tf.keras.layers.Flatten()(bnd_input)
    bnd_x = tf.keras.layers.Dense(66)(bnd_x)
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
        x = UpBlock(width, k, block_depth)([x, skips])

    output = create_output_model(x, widths, k)

    # Create model
    model = Model(inputs=[img_input, bnd_input], outputs=[output])

    return model


# %%
# Train model in a session
gpus = tf.config.list_logical_devices("GPU")
strategy = tf.distribute.MirroredStrategy(
    devices=gpus, cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
)
with strategy.scope():
    model = cnn_model(raster_shape, (22, 3), widths, kernel_size, block_depth)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.9
    )
    opt = Adam(learning_rate=lr_schedule)

    model.compile(
        loss="mean_squared_error",
        optimizer=opt,
        steps_per_execution=steps_per_execution,
    )

    log_dir = str(model_dir / ("logs/fit/" + model_version))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1
    )
    # prediction_callback = PredictionHistory()
    # debug_callback = DebugCallback()

    model.fit(
        x=train_dataset,
        validation_data=test_dataset,
        epochs=n_epochs,
        verbose=2,
    )  # , prediction_callback])

# model.save(model_dir / (model_version + ".h5"))
