# TurboSWAN: Developer Guide

This guide covers maintaining the integrated TurboSWAN project under
`ai-platform-mlops/projects/turboswan/`. For preparing data, running experiments,
editing recipes, and interpreting recorded results, see [README.md](README.md).

Run the commands below from the `ai-platform-mlops` repository root, using its
existing `.venv`. TurboSWAN is a project in that repository, not a separate clone.
Keep project changes under `projects/turboswan/`; coordinate changes to shared
SDK behavior, scripts, workflows, and dependency constraints with the platform
owners. Scientific model architecture, data meaning, and acceptance criteria
remain with the TurboSWAN researchers.

## Project Structure

- `src/turboswan/training.py`: configuration, data checks, dataset creation, model fitting, and optional local Keras outputs.
- `src/turboswan/training_mlflow.py`: one training run recorded in MLflow.
- `src/turboswan/training_sweep.py`: a parameter sweep over tracked training runs, locally or on Ray.
- `src/turboswan/core/`: the researcher-owned model architecture, dataset pipeline, and preprocessing.
- `src/turboswan/mlops/tracking/`: MLflow records and researcher-facing artifacts.
- `src/turboswan/mlops/sweeps/`: sweep configuration, proposal, execution, and coordination.
- `notebooks/`: tracked training and sweep demonstrations.
- `tests/`: focused tests for package structure, tracking, and sweep behavior.

Changing the model architecture or data meaning is a scientific change and should
be agreed with the TurboSWAN researchers.

## Developer Checks

### Linting

Lint only TurboSWAN with the pinned standalone tool:

```bash
uvx ruff==0.14.0 check projects/turboswan/
```

Ruff checks Python source, tests, DAGs, and notebook code without running training
or changing files. The project configuration enables basic error checks, including
undefined names and unused imports, and excludes the personal
`notebooks/demo_dnai_notes.ipynb` notebook. Git-ignored files are also excluded.
Lambda-assignment style (`E731`) is not enforced.
This command does not change the repository `.venv` or shared lockfile. No shared
CI workflow or automatic formatting is enabled.

### Type Checking

Check TurboSWAN's Python types with the pinned standalone checker:

```bash
uvx --from 'pyright[nodejs]==1.1.407' \
  --with nodejs-wheel-binaries==22.20.0 \
  pyright --project projects/turboswan/pyproject.toml
```

The project configuration checks `src/` and `tests/` in basic mode for Python
3.12, resolving dependencies from the existing repository `.venv`. This command
does not modify source files, the training environment, or the shared lockfile.
The standalone tool also installs a pinned Node.js runtime; no system Node.js
installation is required.
It is a local diagnostic check, not a shared CI gate. Findings require review;
third-party type information can be incomplete. Pyright CLI does not check
notebook cells; use Pylance in the editor and notebook execution tests for those.

### Tests

Run the focused regression suite with the existing repository environment:

```bash
.venv/bin/python -m pytest projects/turboswan/tests -q
```

Real-data training tests remain opt-in through the `e2e` marker. Prepare the data
described in the [getting started guide](https://potential-doodle-8gm7w41.pages.github.io/turboswan-mlx/demo-getting-started/), then select those tests explicitly:

```bash
.venv/bin/python -m pytest projects/turboswan/tests -o addopts='' -m e2e -q
```

The fixture uses the folder named by `data.source` in `configs/demo.toml`
(`~/turboswan/data/demo`) and skips tests if required inputs are missing. Skipped tests do not demonstrate
successful training. Model and local MLflow outputs use pytest's temporary paths.

### Developer Check: Fresh Environment and Demo Notebooks

Run this once in a while, not on every commit: before sharing the branch with
researchers, and after changing dependencies, the install steps in the
[getting started guide](https://potential-doodle-8gm7w41.pages.github.io/turboswan-mlx/demo-getting-started/), or the demo notebooks. It checks what a researcher
experiences on a new machine: that the getting-started install steps work and both
`notebooks/demo_tracking.ipynb` and `notebooks/demo_sweep.ipynb` run top to bottom.

```bash
.venv/bin/python -m pytest projects/turboswan/tests -o addopts='' -m developer -q
```

It takes a few minutes with a warm uv cache; a cold cache downloads TensorFlow and
the other dependencies. The test:

1. copies the repository files from disk, including uncommitted edits, into a
   temporary folder with its own home directory;
2. runs the getting-started steps there (`uv venv`, `uv sync --package turboswan`, kernel
   registration) with a minimal `PATH`;
3. runs both notebooks with the `turboswan` kernel they request, including a local
   Ray sweep.

It needs `uv` and the demo data described in the
[getting started guide](https://potential-doodle-8gm7w41.pages.github.io/turboswan-mlx/demo-getting-started/). Your real
`~/turboswan`, Jupyter kernels, and `.venv` are not changed, although the shared uv
cache is added to. The temporary folder, its new environment, kernel, MLflow records,
and Ray session are deleted afterwards, even on failure. Set
`TURBOSWAN_KEEP_DEV_ENV=1` to keep the folder for inspection. A pass means the
notebooks ran without errors; it does not check scientific results.

### Coverage

Install the pinned coverage tools into the existing repository environment:

```bash
uv pip install --python .venv/bin/python pytest-cov==7.0.0 coverage==7.10.7
```

These are local development tools, not training dependencies. This install does
not update project dependencies or the shared lockfile; repeat it after recreating
or synchronizing an environment that removes the tools.

Run the lightweight suite with line and branch coverage:

```bash
.venv/bin/python -m pytest projects/turboswan/tests -q \
  --cov=turboswan --cov-config=projects/turboswan/pyproject.toml \
  --cov-report=term-missing --cov-report=html --cov-report=xml
```

The report measures only the `turboswan` package, not tests or the shared SDK.
The terminal lists uncovered lines and branch destinations; open
`projects/turboswan/htmlcov/index.html` for the annotated source view. Raw coverage
data, HTML, and XML reports are generated under `projects/turboswan/` and ignored
by Git. Normal pytest runs do not collect coverage.

The default excludes real-data tests, so this baseline does not demonstrate
end-to-end training or scientific correctness. No minimum percentage or shared
CI gate is enforced yet. A covered line was executed, not necessarily checked
by a meaningful assertion; branch coverage also records which conditional paths
were taken. Notebook cells and separate Ray worker processes are not included.

## Historical README Fragment

The fragment below was present before this documentation split. It describes the
older `train-network/` layout and ends mid-description. It is preserved as
historical context, not as verified instructions for the integrated package.
Use [README.md](README.md) for the current training entry points.

<details>
<summary>Show the older README fragment</summary>

### TurboSwan

TurboSwan contains the code used to train TensorFlow models that emulate SWAN wave-model output from gridded forcing data and boundary conditions. The repository is centered on the training pipeline in `train-network/`, with supporting scripts for data preparation and notebooks for analysis and validation.

### What This Repository Does

The project trains convolutional neural networks on TFRecord datasets derived from SWAN simulations. In the current setup, the main model predicts wave height (`hs`) on a `(481, 421)` grid from:

- gridded input fields for wind and water level
- boundary-condition for wave height, wave period and wave angle sampled along the model boundary
- normalization metadata stored in NetCDF files
- a validity mask to exclude land mass, used inside the loss function

The codebase assumes an offline workflow where TFRecords have already been generated before model training starts.

### Installation

The project is defined with `pyproject.toml` and targets Python 3.12+.

```bash
uv sync
```

Use `uv run` to execute project commands inside the managed environment.

### Main Training Workflow

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

### The `train-network/` Folder

This folder contains the core implementation of the learning pipeline.

#### `train_network.py`

This is the executable training script. It wires together dataset loading, normalization metadata, multi-GPU TensorFlow strategy setup, model compilation, checkpointing, TensorBoard logging, and final model export.

Notable characteristics of the current implementation:

- it is organized around TFRecord directories such as `train_data` and `val_data`
- it reads the TFRecord root directory and core hyperparameters from a TOML config file
- it loads a `valid-mask.npy` file and appends that mask to the training targets
- it copies the contents of `train-network/` into the output model directory for experiment traceability
- it currently uses `TurboSwanModel.py`, which uses both raster inputs and boundary inputs

#### `create_datasets.py`

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

#### `preprocessing_utils.py`

This module contains lower-level preprocessing helpers used during dataset creation. It provides:

- min/max normalization
- NaN cleanup
- channel reshaping for temporal slices
- wind-stress related helper functions used to derive normalized auxiliary variables

#### `TurboSwanModel.py`

This file contains the main model architecture used by `train_network.py`.

The model combines:

- a convolutional encoder-decoder for 2D raster inputs

</details>