# TurboSWAN: Researcher Guide

TurboSWAN trains TensorFlow models that emulate SWAN wave-model output from gridded forcing data and boundary conditions. It predicts fields such as significant wave height (`hs`) on a fixed `(481, 421)` grid.

The training data is prepared before this project runs. A recipe selects TFRecord training and validation shards, normalization ranges from NetCDF, and a spatial validity mask used by the loss.

This guide covers running experiments with the integrated project in
`ai-platform-mlops`. For code structure, tests, linting, type checking, and
coverage, see [README-developers.md](README-developers.md).

## Run a Tracked Experiment

For a local demonstration, follow the
[getting started guide](https://potential-doodle-8gm7w41.pages.github.io/turboswan-mlx/demo-getting-started/)
(MLX documentation; access requires the Deltares-research organization). It
covers getting this branch, installing the environment, choosing the notebook
kernel, and copying the demo data to `~/turboswan/data/demo`.

Then open `notebooks/demo_tracking.ipynb` and run it top to bottom with its working directory set to `projects/turboswan/notebooks`. This standalone local demo introduces the research workflow without requiring a Data & AI platform account. It uses `configs/demo_mlflow_local.toml`, which keeps the training settings of `configs/demo.toml` and requests an error map from one demo validation example. A local record is stored under `~/turboswan/mlflow/`.

The notebook logs the recipe, input manifests, normalization and mask inputs, learning curve, model signature, and fitted Keras model to MLflow. It also records both the configured and effective runtime data locations so local runs remain reproducible.

After training, the fitted model and input manifests are saved before the
comparison and preprocessing reports are generated. If report generation
fails, the error still propagates and the run is marked failed, but the
already-logged model remains available in MLflow.

### Why Save TOML and JSON?

`metadata/recipe.toml` preserves the original recipe as loaded, including comments
and formatting. `metadata/config.json` records the effective settings, including
defaults and notebook or sweep overrides, with the path formatting described
below. Keep the TOML to understand the starting experiment and the JSON to see
what was actually used. Sweep trials share the base recipe but have their own
effective JSON. Configurations built directly in Python without a source recipe
log JSON only; an original TOML is not invented.

### Edit a Recipe

Copy `configs/demo.toml` and edit the copy. What every setting means, how a sweep
file refers to a recipe, the readable MLflow labels, and the optional comparison
plots are explained in the
[TOML configuration page](https://potential-doodle-8gm7w41.pages.github.io/turboswan-mlx/toml-configuration/)
(MLX documentation; access requires the Deltares-research organization).
Labels and comparisons only change what is displayed or plotted, never the
training calculations or which sweep trial is selected.

### Recorded Data Locations

Local home-directory paths are recorded with `~`, not an expanded username. For
example, Parameters shows `Data - source = ~/turboswan/data/demo`,
`Data - training split = train_data`, and `Data - validation split = val_data`.
Each comparison names a folder inside that source (`subdir`; `.` means the
source itself), with the selected shard recorded by filename. Comparisons are
recorded individually, not also as a JSON-valued parameter.

In `dnai` mode the data folder is likewise recorded as a shortened local path. Dataset manifests and their digests
use shard filenames, so moving the same named shards does not change that digest;
this is a filename-manifest digest, not a checksum of file contents.

`metadata/train-tfrecords.json` and `metadata/val-tfrecords.json` are native
MLflow tables, logged with `mlflow.log_table`. Their `tfrecord_file` column lists
the sorted shard filenames selected after applying `max_tfrecords`, and
`file_path` records each file's location, abbreviating home-directory paths
with `~`. The
corresponding dataset input records hold each split's folder and filename digest.
Together these identify the selected files without copying TFRecord contents.
The tables can be read back with `mlflow.load_table`.

In the MLflow UI, open **Artifacts -> metadata -> train-tfrecords.json** or
**val-tfrecords.json** to view these tables. The **Datasets** panel shows each
split's folder, digest and table schema, not the individual file references.
Its row count is the number of selected shards, not the number of training
examples. A shard is one TFRecord file containing part of the dataset.

We keep one dataset input per split rather than one per shard to avoid a long
dataset list for larger runs. The native JSON tables retain the exact selected
file references separately from the recipe and effective configuration.
They do not preserve data versions: changing a file's contents without renaming
it does not change the filename digest, and the original data must still be
retained outside MLflow.

The complete `metadata/config.json` snapshot also abbreviates home-directory
path fields, while retaining defaults, labels, and overrides.
Explicit absolute paths outside the user's home (for example `/tmp` or a shared
mount) remain explicit; they cannot safely be presented as home-relative paths.
Python still opens files using the original runtime paths. MLflow's storage URIs
are unchanged. No files are moved, and `~` does not make data available on another
machine: that user still needs the corresponding files in their data root.

## Recipes and Direct Training

Recipes in `configs/` define the data layout, augmentation, model configuration, fitting schedule, and optional local outputs. Copy `configs/demo.toml` before changing a research question.

To run one recipe without MLflow tracking:

```bash
.venv/bin/python -m turboswan.training --config projects/turboswan/configs/demo.toml
```

To run the same recipe as one MLflow run, using its `[mlflow]` section:

```bash
.venv/bin/python -m turboswan.training_mlflow --config projects/turboswan/configs/demo.toml
```

To run a sweep over a recipe, recorded as one MLflow parent run with a child run per trial:

```bash
.venv/bin/python -m turboswan.training_sweep --sweep-config projects/turboswan/configs/sweeps/demo.toml --executor local
```

Use `--executor ray` to run trials on the active Ray cluster.

The same sweep is available interactively in `notebooks/demo_sweep.ipynb` (kernel **Python (turboswan)**, working directory `projects/turboswan/notebooks`). It starts a local Ray instance on your machine and runs three trials of the demo data; this takes a couple of minutes on a laptop-class CPU. It needs the same demo data in `~/turboswan/data/demo` and records to `~/turboswan/mlflow/`.

The standard data layout is:

```text
<data.source>/
	train_data/
	val_data/
	min_max_files/min-max-values-all.nc
	valid-mask.npy
```

`data.source` is always a folder that already contains the data; nothing is downloaded, and training raises `FileNotFoundError` if the folder does not exist. One setting, `[run] mode`, decides where MLflow records:

| `[run] mode` | MLflow records to |
|---|---|
| `"local"` (default) | a local folder (`mlflow.local_tracking_dir`) |
| `"dnai"` | the Data & AI platform's MLflow server |

`mlflow.local_tracking_dir` is rejected in dnai mode with an error.

Changing the model architecture or data meaning is a scientific change and should be agreed with the TurboSWAN researchers.
