# %%
import json
import re
import pathlib
import multiprocessing as mp
import xarray as xr
import tqdm

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# %%
src_path = pathlib.Path('/gpfs/work2/0/prjs1027/swn-runs/02-2022/computations')
boundary_path = pathlib.Path('/gpfs/work2/0/prjs1027/guus/data/BoundaryConditions/bnd.nc')
write_dir = pathlib.Path('/gpfs/work2/0/prjs1027/tfrecords/run2')
write_dir.mkdir(exist_ok=True, parents=True)
filtered_paths = [path for path in src_path.glob('ens*') if re.search(r'\d+$', str(path))]

# %%
boundary_ds = xr.open_dataset(boundary_path)

# %%
output_variables = ["hs", "theta0_x", "theta0_y", "tmm10", "tps", "hswe"]
input_variables = ["xwnd", "ywnd", "ssh", "xcur", "ycur"]
boundary_variables = ["hs", "hswe", "tmm10"]
conversion_dict = {"hs": "Hm0", "hswe": "HE10", "tmm10": "Tmm10"}

# %%
def compute_min_max(path):
    all_datasets = []
    
    for _ in tqdm.tqdm(list(path.glob('*'))):
        try:
            ds_path = next(_.glob('output/*.nc'))
            ds = xr.open_dataset(ds_path)
            all_datasets.append(ds)
        except:
            continue

    combined_ds = xr.concat(all_datasets, dim="time")
    min_max_values = {}
    for var in combined_ds.variables:
        min_value = combined_ds[var].min().values
        max_value = combined_ds[var].max().values
        min_max_values[var] = np.array([min_value, max_value])
    return min_max_values

# %%
min_max_values = {}
for paths in filtered_paths:
    min_max_values[paths.stem] = compute_min_max(paths)

# %%
min_max_values_detailed = {}
for ens in min_max_values.keys():
    for var in min_max_values[ens].keys():
        try:
            min_max_values_detailed[var] = np.concat([min_max_values_detailed[var], min_max_values[ens][var]])
        except:
            min_max_values_detailed[var] = min_max_values[ens][var]

# %%
for var in min_max_values_detailed.keys():
    min_value = min_max_values_detailed[var].min()
    max_value = min_max_values_detailed[var].max()
    min_max_values_detailed[var] = {
        "minimum": min_value,
        "maximum": max_value
    }

# %%
df = pd.DataFrame(min_max_values_detailed)

# %%
ds_min_max = xr.Dataset.from_dataframe(df)
ds_min_max_dir = write_dir / "min-max-values"
ds_min_max_dir.mkdir(exist_ok=True, parents=True)
ds_min_max.to_netcdf(ds_min_max_dir / "min-max-values.nc")


