# TurboSwan

TurboSwan contains the code used to train TensorFlow models that emulate SWAN wave-model output from gridded forcing data and boundary conditions. The repository is centered on the training pipeline in `train-network/`, with supporting scripts for data preparation and notebooks for analysis and validation.

## What This Repository Does

The project trains convolutional neural networks on TFRecord datasets derived from SWAN simulations. In the current setup, the main model predicts wave height (`hs`) on a `(481, 421)` grid from:

- gridded input fields for wind and water level
- boundary-condition for wave height, wave period and wave angle sampled along the model boundary
- normalization metadata stored in NetCDF files
- a validity mask to exclude land mass, used inside the loss function

The codebase assumes an offline workflow where TFRecords have already been generated before model training starts.

## Installation

The project is defined with `pyproject.toml` and targets Python 3.12+.

```bash
uv sync
```

Use `uv run` to execute project commands inside the managed environment.

## Main Training Workflow

The primary entry point is `train-network/train_network.py`.

At a high level, the training flow is:

1. Define the training configuration, including TFRecord locations, model version, variables, batch size, and network width/depth.
2. Load min/max normalization values from a NetCDF dataset.
3. Build TensorFlow datasets from TFRecord files for training and validation.
4. Construct the TurboSwan model.
5. Compile the model with a masked mean-squared-error loss.
6. Train with TensorBoard logging and model checkpoints.
7. Save the final trained model to `trained_models/<model_version>/`.

Before running training, create a local config file from the example and update it for your environment.

```bash
cp train-network/config.example.toml train-network/config.toml
```

The config file is used to provide the TFRecord root directory and the main training settings:

- `model_version`
- `tfrecord_dir`
- `learning_rate`
- `n_epochs`
- `batch_size`
- `seed`

Example:

```bash
uv run python train-network/train_network.py
```

To use a different config file:

```bash
uv run python train-network/train_network.py --config path/to/config.toml
```

## The `train-network/` Folder

This folder contains the core implementation of the learning pipeline.

### `train_network.py`

This is the executable training script. It wires together dataset loading, normalization metadata, multi-GPU TensorFlow strategy setup, model compilation, checkpointing, TensorBoard logging, and final model export.

Notable characteristics of the current implementation:

- it is organized around TFRecord directories such as `train_data` and `val_data`
- it reads the TFRecord root directory and core hyperparameters from a TOML config file
- it loads a `valid-mask.npy` file and appends that mask to the training targets
- it copies the contents of `train-network/` into the output model directory for experiment traceability
- it currently uses `TurboSwanModel.py`, which uses both raster inputs and boundary inputs

### `create_datasets.py`

This module is responsible for turning TFRecord files into batched `tf.data.Dataset` pipelines.

Its responsibilities include:

- discovering TFRecord shards in a directory
- parsing serialized tensors from each example
- normalizing variables with min/max metadata
- reshaping raster channels and boundary-condition inputs
- concatenating inputs and outputs into the structure expected by the model
- optionally appending the spatial validity mask to the target tensor
- optionally applying data augmentation such as flips and rotations

The output format matches the model signature used in training:

- image inputs for gridded variables
- one-dimensional boundary inputs for along-boundary forcing
- output tensors with an additional mask channel used by the loss

### `preprocessing_utils.py`

This module contains lower-level preprocessing helpers used during dataset creation. It provides:

- min/max normalization
- NaN cleanup
- channel reshaping for temporal slices
- wind-stress related helper functions used to derive normalized auxiliary variables

### `TurboSwanModel.py`

This file contains the main model architecture used by `train_network.py`.

The model combines:

- a convolutional encoder-decoder for 2D raster inputs
- residual blocks in both the downsampling and upsampling paths
- a dense branch for boundary-condition inputs
- a fusion stage that merges raster and boundary representations before reconstruction
- a masked loss that ignores invalid grid cells during optimization

Architecturally, it behaves like a residual U-Net variant with an additional boundary-input branch.

### `model_class.py`

This file contains an earlier model variant. It implements a similar residual encoder-decoder structure, but without the explicit boundary-input branch found in `TurboSwanModel.py`.

## Data Expectations

The training code assumes that data preparation has already produced:

- TFRecord files containing serialized tensors for all configured variables
- a NetCDF file with min/max values for normalization
- a NumPy validity mask file

Given a configured `tfrecord_dir`, the training script currently expects at least:

- `train_data/`
- `val_data/`
- `valid-mask.npy`
- `min_max_files/min-max-values-all.nc`

In the current training script, the configured variables are grouped into:

- gridded inputs `xwnd`, `ywnd`, and `wl`
- outputs `hs`
- boundary variables `hs_bnd`, `tp_bnd`, and `theta0_bnd`

The preprocessing code also assumes fixed spatial dimensions and specific boundary lengths, so dataset generation and model training must stay aligned on shape conventions.

## Outputs

Training artifacts are written to a model-specific directory under `trained_models/`. These artifacts typically include:

- TensorBoard logs
- best-checkpoint model weights
- the final saved model
- a copy of the training source files used for that run

## Scripts

The `scripts/` folder contains utility programs for dataset generation, min/max computation, experiment execution, TFRecord writing, and hyperparameter tuning. These scripts support the broader workflow around model training, but they are separate from the core training implementation in `train-network/`.

## Notebooks

The `notebooks/` folder contains exploratory, validation, and analysis notebooks. These are useful for inspecting datasets, reviewing model predictions, generating validation material, and experimenting interactively, but they are not the primary implementation surface for the training pipeline.

