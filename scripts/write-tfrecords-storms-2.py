import datetime
import json
import multiprocessing as mp
import os
import pathlib
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
from dateutil.relativedelta import relativedelta

import numpy.ma as ma
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm
import xarray as xr
import random


src_path = pathlib.Path('/gpfs/work2/0/prjs1027/swn-runs')
boundary_path = pathlib.Path(
    "/gpfs/work2/0/prjs1027/guus/data/BoundaryConditions/bnd_storms.nc"
)

bathymetry_path = pathlib.Path('/gpfs/work2/0/prjs1027/swn-runs/02-2022/geometry/swan-ns-j22_6-v1a_adjust.bot')
write_dir = pathlib.Path("/gpfs/work2/0/prjs1027/tfrecords/run3")
write_dir.mkdir(exist_ok=True, parents=True)
filtered_paths = [
    path for path in src_path.glob("ens*") if re.search(r"\d+$", str(path))
]
filtered_paths = [path for path in src_path.glob('0*') if path.stem[-4:] != '2022']
# ens = os.environ.get("ENS")
ens = sys.argv[1]
# filtered_path = src_path / f"ens{ens}"
# print(filtered_path)

boundary_ds = xr.open_dataset(boundary_path)

output_variables = ["hs", "theta0", "tmm10", "tps", "hswe"]
input_variables = ["xwnd", "ywnd", "wl", "xcur", "ycur"]
boundary_variables = ["hs", "hswe", "tmm10", "theta0"]
conversion_dict = {"hs": "Hm0", "hswe": "HE10", "tmm10": "Tmm10", "theta0": "meanD"}

random.seed(42)
random.shuffle(filtered_paths)

train_storms = filtered_paths[:5]
val_storms = list([filtered_paths[5]])
test_storms = list([filtered_paths[6]])

print(f"Training storms: {train_storms}")
print(f"Validation storms: {val_storms}")
print(f"Testing storms: {test_storms}")

def load_ascii_bathy(input_grid_file_path, lon_start, lat_start, lon_step, lat_step):
    """
    Loads bathymetry and grid from ascii file (GEBCO)
 
    Args:
        input_grid_file_path (str): path to grid file
        lon_start: start longitude
        lat_start:
    Returns:
        lon, lat, bottom_data (np.array): 2D arrays with lon and lat values
        and bottom data
 
    Example usage:
        lon_start, lat_start, lon_step, lat_step = -3.000000, 51.500000, 0.500000, 0.500000
        lon_bathy, lat_bathy, bottom_data = input_loader.load_bathy(extra_path + "depth.GRD", lon_start, lat_start, lon_step, lat_step)
 
    """
    # Read the bottom grid data from the ASCII file
    bottom_data = np.loadtxt(input_grid_file_path)
 
    # Transpose the data to match the provided organization (latitude columns, longitude rows)
    bottom_data = bottom_data.T
 
    # Replace -999 with NaN
    bottom_data[bottom_data == -999] = np.nan
 
    lons = np.arange(lon_start, lon_start + len(bottom_data[0]) * lon_step, lon_step)
    lats = np.arange(lat_start, lat_start + len(bottom_data) * lat_step, lat_step)
 
    # Create a meshgrid from the coordinates
    lon, lat = np.meshgrid(lons, lats)
 
    return lon, lat, bottom_data

bathymetry = load_ascii_bathy(bathymetry_path, -12, 48, 0.05, 1/30)[2]

def correct_nan_theta0(ds):
    for i in range(ds.time.shape[0]):
        hs = ds['hs'].values[i, ...]
        theta0 = ds['theta0'].values[i, ...]
        mask_hs = ma.masked_invalid(hs).mask
        mask_theta0 = ma.masked_invalid(theta0).mask
        
        invalid_ns_mask = np.logical_xor(mask_hs, mask_theta0)
        if invalid_ns_mask.sum() > 0:
            # Extract valid data points
            x = np.arange(theta0.shape[1])
            y = np.arange(theta0.shape[0])
            
            xx, yy = np.meshgrid(x, y)
            x1 = xx[~invalid_ns_mask]
            y1 = yy[~invalid_ns_mask]
            
            new_data = theta0[~invalid_ns_mask]
                    
            # Create interpolator for valid data
            interpolated = griddata((x1, y1), new_data.ravel(), (xx, yy), method='nearest', fill_value=None)
            ds['theta0'].values[i, ...] = interpolated
    return ds

def correct_outlier_cur(ds):
    for i in range(ds.time.shape[0]):
        xcur = ds['xcur'].values[i, ...]
        ycur = ds['ycur'].values[i, ...]
        
        outliers_xcur = np.abs(xcur) > 3
        outliers_ycur = np.abs(ycur) > 3
        
        x = np.arange(xcur.shape[1])
        y = np.arange(xcur.shape[0])

        xx, yy = np.meshgrid(x, y)
        x1_xcur = xx[~outliers_xcur]
        y1_xcur = yy[~outliers_xcur]

        x1_ycur = xx[~outliers_ycur]
        y1_ycur = yy[~outliers_ycur]

        new_xcur = xcur[~outliers_xcur]
        new_ycur = ycur[~outliers_ycur]

        interpolated_xcur = griddata((x1_xcur, y1_xcur), new_xcur.ravel(), (xx, yy), method='nearest', fill_value=None)
        interpolated_ycur = griddata((x1_ycur, y1_ycur), new_ycur.ravel(), (xx, yy), method='nearest', fill_value=None)
        
        ds['xcur'].values[i, ...] = interpolated_xcur
        ds['ycur'].values[i, ...] = interpolated_ycur

    return ds

def gather_input(path, boundary_ds, input_variables, output_variables, boundary_variables):
    variables = input_variables + output_variables

    input_fields = {}
    input_fields = {var: [] for var in variables}
    boundary_fields = {var: [] for var in boundary_variables}
    for _ in tqdm.tqdm(list(path.glob("*"))):
        year = _.stem[:4]
        month = _.stem[4:6]
        year_month = f'{year}-{month}'
        start_date = datetime.datetime.strptime(year_month, '%Y-%m')
        if year == '2013':
            end_date = (start_date + relativedelta(days=6)) - relativedelta(minutes=1)
        else:
            end_date = (start_date + relativedelta(months=1)) - relativedelta(minutes=1)
        try:
            ds_path = next(_.glob("output/*.nc"))
            ds = xr.open_dataset(ds_path)
            ds = ds.sel(time=slice(start_date, end_date))
            ds = ds.resample(time='6H').nearest()
        except:
            # print(f"{_ / 'output/swan2d.nc'} does not exist, skipping")
            continue

        ds['wl'] = ds['ssh'] + bathymetry
        ds = correct_nan_theta0(ds)
        ds = correct_outlier_cur(ds)

        for var in variables:
            data = ds[var].values
            shapes = data.shape

            if var in input_variables:
                combined_fields = np.empty((shapes[0] - 2, shapes[1], shapes[2], 3))
                for t in range(2, shapes[0]):
                    combined_fields[t - 2, ..., 0] = data[t - 2, ...]
                    combined_fields[t - 2, ..., 1] = data[t - 1, ...]
                    combined_fields[t - 2, ..., 2] = data[t, ...]
            elif var in output_variables:
                combined_fields = np.empty((shapes[0] - 2, shapes[1], shapes[2], 1))
                for t in range(2, shapes[0]):
                    combined_fields[t - 2, ..., 0] = data[t, ...]
                    
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

# Define the function to process each path
def process_path(paths, write_dir):
    for path in paths:
        ens_path = path / f'computations/ens{ens}'
        print(f'Processing path {ens_path}')
        variable_fields, boundary_fields = gather_input(
            ens_path,
            boundary_ds=boundary_ds,
            input_variables=input_variables,
            output_variables=output_variables,
            boundary_variables=boundary_variables,
        )
        for var in boundary_variables:
            variable_fields[f"{var}_bnd"] = boundary_fields[var]

        write_dir.mkdir(exist_ok=True, parents=True)
        write_data(
            variables=variable_fields,
            filename=f'storm_{path.stem}_{ens_path.stem}',
            max_files=10,
            out_dir=str(write_dir) + "/",
        )
            # print(f'Path {path} passed an error')
        print(f"Processed {ens_path}")



if __name__ == "__main__":
    process_path(test_storms, write_dir=write_dir / 'test_data')
    process_path(val_storms, write_dir=write_dir / 'val_data')
    process_path(train_storms, write_dir=write_dir / 'train_data')
