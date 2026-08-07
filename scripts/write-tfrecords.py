# %%
import json
import multiprocessing as mp
import os
import pathlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
import xarray as xr

# %%
src_path = pathlib.Path("/gpfs/work2/0/prjs1027/swn-runs/02-2022/computations")
boundary_path = pathlib.Path(
    "/gpfs/work2/0/prjs1027/guus/data/BoundaryConditions/bnd.nc"
)
write_dir = pathlib.Path("/gpfs/work2/0/prjs1027/tfrecords/run2")
write_dir.mkdir(exist_ok=True, parents=True)
filtered_paths = [
    path for path in src_path.glob("ens*") if re.search(r"\d+$", str(path))
]
# ens = os.environ.get("ENS")
ens = sys.argv[1]
filtered_path = src_path / f"ens{ens}"
print(filtered_path)

# %%
boundary_ds = xr.open_dataset(boundary_path)

# %%
output_variables = ["hs", "theta0_x", "theta0_y", "tmm10", "tps", "hswe"]
input_variables = ["xwnd", "ywnd", "ssh", "xcur", "ycur"]
boundary_variables = ["hs", "hswe", "tmm10"]
conversion_dict = {"hs": "Hm0", "hswe": "HE10", "tmm10": "Tmm10"}


# %%
def gather_input(path, boundary_ds, variables, boundary_variables):
    input_fields = {}
    input_fields = {var: [] for var in variables}
    boundary_fields = {var: [] for var in boundary_variables}
    for _ in tqdm.tqdm(list(path.glob("*"))):
        try:
            ds_path = next(_.glob("output/*.nc"))
            ds = xr.open_dataset(ds_path)
        except:
            # print(f"{_ / 'output/swan2d.nc'} does not exist, skipping")
            continue

        for var in variables:
            if var == "theta0_x":
                data = np.cos(np.deg2rad(ds["theta0"].values))
            elif var == "theta0_y":
                data = np.sin(np.deg2rad(ds["theta0"].values))
            else:
                data = ds[var].values

            shapes = data.shape
            combined_fields = np.empty((shapes[0], shapes[1], shapes[2], 2))
            for t in range(shapes[0]):
                combined_fields[t, ..., 0] = data[t - 1, ...]
                combined_fields[t, ..., 1] = data[t, ...]

                if var in list(boundary_fields.keys()):
                    boundary_field = (
                        boundary_ds[conversion_dict[var]].sel(time=ds.time[t]).values
                    )
                    boundary_field = boundary_field.reshape(
                        (1, boundary_field.shape[0])
                    )
                    boundary_fields[var].append(boundary_field.astype(np.float32))
            input_fields[var].append(combined_fields.astype(np.float32))

    # Concatenate all the arrays for each variable
    for var in variables:
        input_fields[var] = np.concatenate(input_fields[var], axis=0)
        if var in list(boundary_fields.keys()):
            boundary_fields[var] = np.concatenate(boundary_fields[var], axis=0)

    print(f"Done with {path}")
    return input_fields, boundary_fields


# %%
def get_files(data_path):
    files = tf.io.gfile.glob(data_path + "/" + "*.tfrecords")
    return files


def get_dataset(files):
    """return a tfrecord dataset with all tfrecord files"""
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(tf_parse)
    return dataset


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_combined_data(variables):
    # define the dictionary -- the structure -- of our single example
    var = list(variables.keys())[0]
    data = {
        "height": _int64_feature(variables[var].shape[0]),
        "width": _int64_feature(variables[var].shape[1]),
        "depth": _int64_feature(variables[var].shape[2]),
    }

    # define dictionary for each mode
    for var in variables.keys():
        data[var] = _bytes_feature(serialize_array(variables[var]))

    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def write_data(variables, filename, max_files, out_dir):
    """Writes the data to multiple tfrecord files each containing max_files examples"""
    try:
        filename = filename.stem
    except:
        pass
    var = list(variables.keys())[0]
    splits = (len(variables[var]) // max_files) + 1
    if len(variables[var]) % max_files == 0:
        splits -= 1

    print(
        f"\nUsing {splits} shard(s) for {len(variables[var])} files,\
            with up to {max_files} samples per shard"
    )

    file_count = 0

    for i in tqdm.tqdm(range(splits)):
        if i == splits - 1 and len(variables[var]) % max_files != 0:
            current_shard_name = "{}{}_{}_{}_{}.tfrecords".format(
                out_dir, i + 1, splits, filename, len(variables[var]) % max_files
            )
        else:
            current_shard_name = "{}{}_{}_{}_{}.tfrecords".format(
                out_dir, i + 1, splits, filename, max_files
            )

	# options = tf.io.TFRecordOptions(compression_type="ZLIB")
        writer = tf.io.TFRecordWriter(current_shard_name)#, options=options)

        current_shard_count = 0
        while current_shard_count < max_files:
            index = i * max_files + current_shard_count
            if index == len(variables[var]):
                break

            current_variables = {}
            for key in variables.keys():
                current_variables[key] = variables[key][index]

            out = parse_combined_data(current_variables)

            writer.write(out.SerializeToString())
            current_shard_count += 1
            file_count += 1

        writer.close()

    print(f"\nWrote {file_count} elements to TFRecord")
    return file_count


def get_dataset_large(
    tfr_dir=str(write_dir) + "train_data/", pattern: str = "*train_data.tfrecords"
):
    """Loads the tfrecord files and returns a tfrecord dataset"""
    files = glob.glob(tfr_dir + pattern, recursive=False)

    dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.map(tf_parse)

    return dataset


def tf_parse(eg):
    """parse an example (or batch of examples, not quite sure...)"""

    # here we re-specify our format
    # you can also infer the format from the data using tf.train.Example.FromString
    # but that did not work
    vars = variable_fields
    parse_dict = {
        "height": tf.io.FixedLenFeature([], tf.int64),
        "width": tf.io.FixedLenFeature([], tf.int64),
        "depth": tf.io.FixedLenFeature([], tf.int64),
    }

    for var in vars:
        parse_dict[var] = tf.io.FixedLenFeature([], tf.string)

    example = tf.io.parse_example(
        eg[tf.newaxis],
        parse_dict,
    )
    hs = tf.io.parse_tensor(example["hs"][0], out_type="float32")
    hs_bnd = tf.io.parse_tensor(example["hs_bnd"][0], out_type="float32")
    # inputs = tf.io.parse_tensor(example["inputs"][0], out_type="float64")
    # outputs = tf.io.parse_tensor(example["outputs"][0], out_type="float64")
    return hs, hs_bnd


# %%
# Define the function to process each path
def process_path(path):
    variable_fields, boundary_fields = gather_input(
        path,
        boundary_ds=boundary_ds,
        variables=input_variables + output_variables,
        boundary_variables=boundary_variables,
    )
    for var in boundary_variables:
        variable_fields[f"{var}_bnd"] = boundary_fields[var]
    write_data(
        variables=variable_fields,
        filename=path,
        max_files=10,
        out_dir=str(write_dir) + "/",
    )
    print(f"Processed {path}")


# %%
if __name__ == "__main__":
    # Create a ThreadPoolExecutor with 10 threads
    process_path(filtered_path)
    # with mp.Pool(120) as pool:
    #     pool.map(process_path, filtered_paths)

# %%
