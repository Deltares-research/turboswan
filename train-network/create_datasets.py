import numpy as np

import tensorflow as tf
import xarray as xr
import preprocessing_utils


from typing import Union, List, Optional
from pathlib import Path


def get_files(data_path: str):
    files = tf.io.gfile.glob(data_path + "/" + "*.tfrecords")
    return files


def get_dataset(
    files: list,
    min_max_ds: xr.Dataset,
    variable_fields: List[str],
    input_variables: List[str],
    output_variables: List[str],
    boundary_variables: List[str],
    indices: List[int],
    valid_mask: np.ndarray = None,
):
    """return a tfrecord dataset with all tfrecord files"""
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(
        lambda x: tf_parse(
            x,
            min_max_ds=min_max_ds,
            variable_fields=variable_fields,
            input_variables=input_variables,
            output_variables=output_variables,
            boundary_variables=boundary_variables,
            valid_mask=valid_mask,
            indices=indices,
        )
    )
    return dataset


def tf_parse(
    example_proto,
    min_max_ds: xr.Dataset,
    variable_fields: List[str],
    input_variables: List[str],
    output_variables: List[str],
    boundary_variables: List[str],
    valid_mask: np.ndarray = None,
    indices: List[int] = [1, 2],
):
    # Define the TFRecord schema
    parse_dict = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
    }
    for var in variable_fields:
        parse_dict[var] = tf.io.FixedLenFeature([], tf.string)

    # Parse the example
    example = tf.io.parse_example(example_proto[tf.newaxis], parse_dict)

    # Store computed inputs
    variables = {}

    # --- Parse and process remaining variables ---
    for var in variable_fields:
        tensor = tf.io.parse_tensor(example[var][0], out_type=tf.float32)
        tensor = preprocessing_utils.normalize_and_clean(
            tensor, var, min_max_ds=min_max_ds
        )

        if var in input_variables:
            tensor = preprocessing_utils.reshape_channels(tensor, indices=indices)
        elif var in output_variables:
            tensor = tf.reshape(tensor, (481, 421, 1))
        elif var in boundary_variables:
            tensor = tf.reshape(tensor, (106, 1))

        variables[var] = tensor

    # --- Final input/output tensors ---
    image_inputs = tf.concat([variables[var] for var in input_variables], axis=-1)
    outputs = tf.concat([variables[var] for var in output_variables], axis=-1)
    boundary_inputs = tf.concat([variables[var] for var in boundary_variables], axis=-1)
    if valid_mask is not None:
        valid_mask = tf.convert_to_tensor(valid_mask, dtype=tf.float32)
        valid_mask = tf.reshape(valid_mask, (481, 421, 1))
        outputs = tf.concat([outputs, valid_mask], axis=-1)

    if len(boundary_inputs) > 0:
        inputs = (image_inputs, boundary_inputs)
    else:
        inputs = image_inputs

    return inputs, outputs


def load_min_max_ds(
    file_path: Union[Path, str], boundary_variables: List[str] = []
) -> xr.Dataset:
    min_max_ds = xr.open_dataset(file_path)
    min_max_ds["tp"] = [min_max_ds["tps"].values[0], min_max_ds["tps"].values[1]]
    for bnd_var in boundary_variables:
        min_max_ds[bnd_var] = min_max_ds[bnd_var.split("_")[0]]

    min_tau = preprocessing_utils.compute_wind_stress(
        min_max_ds["xwnd"].values[0], min_max_ds["ywnd"].values[0]
    )
    max_tau = preprocessing_utils.compute_wind_stress(
        min_max_ds["xwnd"].values[1], min_max_ds["ywnd"].values[1]
    )

    min_max_ds["xtau"] = [np.cos(min_tau), np.cos(max_tau)]
    min_max_ds["ytau"] = [np.sin(min_tau), np.sin(max_tau)]
    return min_max_ds


def create_dataset(
    file_dir: Union[Path, str],
    min_max_ds: xr.Dataset,
    variable_fields: List[str],
    input_variables: List[str],
    output_variables: List[str],
    boundary_variables: List[str],
    valid_mask: np.ndarray,
    indices: List[int],
    batch_size: int,
    seed=0,
    flip_horizontal: Optional[bool] = False,
    flip_vertical: Optional[bool] = False,
    rotate_180: Optional[bool] = False,
):
    files = get_files(str(file_dir))
    dataset = get_dataset(
        files,
        min_max_ds=min_max_ds,
        variable_fields=variable_fields,
        input_variables=input_variables,
        output_variables=output_variables,
        boundary_variables=boundary_variables,
        indices=indices,
        valid_mask=valid_mask,
    )

    dataset = dataset.shuffle(
        buffer_size=1000, seed=seed, reshuffle_each_iteration=True
    )

    if flip_horizontal or flip_vertical or rotate_180:
        dataset = dataset.map(
            lambda x, y: augment_pair(x, y, flip_horizontal, flip_vertical, rotate_180)
        )

    dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    return dataset


def augment_pair(
    input_image: tf.Tensor,
    output_image: tf.Tensor,
    flip_horizontal: Optional[bool] = False,
    flip_vertical: Optional[bool] = False,
    rotate_90: Optional[bool] = True,
):
    xtau = input_image[..., :2]
    ytau = input_image[..., 2:4]
    other_channels = input_image[..., 4:]

    # Horizontal flip
    if flip_horizontal and tf.random.uniform(()) > 0.5:
        xtau = tf.image.flip_left_right(xtau) * -1
        ytau = tf.image.flip_left_right(ytau)
        output_image = tf.image.flip_left_right(output_image)

    # Vertical flip
    if flip_vertical and tf.random.uniform(()) > 0.5:
        xtau = tf.image.flip_up_down(xtau)
        ytau = tf.image.flip_up_down(ytau) * -1
        output_image = tf.image.flip_up_down(output_image)

    # Stack back the input
    input_image = tf.concat([xtau, ytau, other_channels], axis=-1)

    # Random 90-degree rotation
    if rotate_90:
        k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        input_image = tf.image.rot90(input_image, k=k)
        output_image = tf.image.rot90(output_image, k=k)

        # Rotate wind vectors accordingly
        x = input_image[..., :2]
        y = input_image[..., 2:4]
        if k == 1:
            x_new = -y
            y_new = x
        elif k == 2:
            x_new = -x
            y_new = -y
        elif k == 3:
            x_new = y
            y_new = -x
        else:
            x_new = x
            y_new = y

        input_image = tf.concat([x_new, y_new, input_image[..., 4:]], axis=-1)

    return input_image, output_image
