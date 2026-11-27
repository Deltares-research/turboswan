"""Tests for turboswan.mlops.tracking."""

from pathlib import Path
from unittest.mock import Mock, patch

import keras
import matplotlib.pyplot as plt
import numpy as np
import pytest
import tensorflow as tf
import xarray as xr

from turboswan import training
from turboswan.mlops.tracking.comparisons import log_comparisons


def test_tracking_labels_load_from_the_training_recipe(tmp_path: Path) -> None:
    recipe_path = tmp_path / "recipe.toml"
    recipe_path.write_text(
        '[data]\nsource = "/tmp/data"\n'
        '[run]\nmodel_dir = "/tmp/models"\nmodel_version = "test"\n'
        '[mlflow]\nexperiment_name = "turboswan/test"\nmodel_name = "turboswan-hs-baseline"\n'
        '[mlflow.metric_labels]\nloss = "Training masked MSE"\n'
        '[mlflow.parameter_labels]\n"training.batch_size" = "Batch size"\n'
        '[mlflow.variable_descriptions]\nhs = "significant wave height"\n'
    )

    config = training.load_config(recipe_path)

    assert config.mlflow is not None
    assert config.mlflow.model_name == "turboswan-hs-baseline"
    assert config.mlflow.metric_labels == {"loss": "Training masked MSE"}
    assert config.mlflow.parameter_labels == {"training.batch_size": "Batch size"}
    assert config.mlflow.variable_descriptions == {"hs": "significant wave height"}


def test_local_tracking_mode_creates_a_sqlite_experiment(tmp_path: Path) -> None:
    from mlflow.tracking import MlflowClient

    from turboswan.mlops.tracking.mlflow import configure_run_tracking

    tracking_uri = configure_run_tracking(
        "turboswan/local-training-test",
        mode="local",
        local_tracking_dir=tmp_path,
    )
    experiment = MlflowClient(tracking_uri).get_experiment_by_name(
        "turboswan/local-training-test"
    )

    assert tracking_uri.startswith("sqlite:///")
    assert experiment is not None


def test_recipe_artifact_preserves_original_toml_and_effective_overrides(tmp_path: Path) -> None:
    from turboswan.mlops.tracking.mlflow import log_preprocessing_artifacts

    recipe_path = tmp_path / "baseline.toml"
    original_recipe = (
        b'# Researcher-selected baseline\r\n[data]\r\nsource = "~/data"\r\n'
        b'[run]\r\nmodel_dir = "~/models"\r\nmodel_version = "baseline"\r\n'
        b'[training]\r\nbatch_size = 2 # starting batch size\r\n'
    )
    recipe_path.write_bytes(original_recipe)
    config = training.load_config(recipe_path).model_copy(deep=True)
    config.training.batch_size = 4
    recipe_path.write_text("# Changed after the recipe was loaded\n")
    artifacts = {}

    def capture_artifact(path, *, artifact_path):
        artifacts[f"{artifact_path}/{Path(path).name}"] = Path(path).read_bytes()

    with (
        patch("turboswan.mlops.tracking.mlflow.mlflow.log_dict") as log_dict,
        patch("turboswan.mlops.tracking.mlflow.mlflow.log_artifact", side_effect=capture_artifact),
        patch("turboswan.mlops.tracking.mlflow.create_preprocessing_artifacts", return_value=[]),
    ):
        log_preprocessing_artifacts(config, Mock())

    assert artifacts["metadata/recipe.toml"] == original_recipe
    snapshot, artifact_path = log_dict.call_args.args
    assert artifact_path == "metadata/config.json"
    assert snapshot["training"]["batch_size"] == 4
    assert snapshot["training"]["n_epochs"] == training.TrainingConfig().n_epochs
    assert set(snapshot) == {"data", "run", "augmentation", "model", "training", "mlflow", "comparisons"}


def test_tracking_metadata_contains_every_validated_config_value() -> None:
    from turboswan import training
    from turboswan.mlops.tracking.metadata import flatten_config

    config = training.TrainConfig.model_validate(
        {
            "data": {"source": "/tmp/data", "output_variables": ["hs", "tp"]},
            "run": {"model_dir": "/tmp/models", "model_version": "test"},
            "augmentation": {"rotate_180": True},
            "training": {"lr_decay_rate": 0.75},
        }
    )

    params = flatten_config(config)

    assert params["data.output_variables"] == '["hs", "tp"]'
    assert params["augmentation.rotate_180"] is True
    assert params["training.lr_decay_rate"] == 0.75


def test_tracking_metadata_records_the_local_data_path_without_the_home_folder(
    tmp_path: Path,
) -> None:
    from turboswan import training
    from turboswan.mlops.tracking.metadata import build_run_metadata

    data_dir = tmp_path / "local-data"
    config = training.TrainConfig.model_validate(
        {
            "data": {"source": str(data_dir)},
            "run": {"model_dir": str(tmp_path / "models"), "model_version": "test"},
        }
    )

    with patch("pathlib.Path.home", return_value=tmp_path):
        metadata = build_run_metadata(config, experiment_name="turboswan/test")

    assert metadata.params["data.source"] == "~/local-data"
    assert metadata.params["run.mode"] == "local"
    assert metadata.params["data.resolved_train_dir"] == config.data.train_subdir
    assert metadata.params["data.resolved_val_dir"] == config.data.val_subdir
    assert config.data.data_dir == data_dir.resolve()


def test_metrics_callback_logs_every_keras_metric() -> None:
    from turboswan.mlops.tracking.mlflow import MetricsCallback

    with patch("turboswan.mlops.tracking.mlflow.mlflow.log_metrics") as log_metrics:
        MetricsCallback().on_epoch_end(
            2,
            {"loss": 0.4, "val_loss": 0.5, "mean_absolute_error": 0.2},
        )

    log_metrics.assert_called_once_with(
        {"loss": 0.4, "val_loss": 0.5, "mean_absolute_error": 0.2},
        step=3,
    )


def test_config_artifact_abbreviates_paths_without_changing_runtime_config(
    comparison_config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from turboswan.mlops.tracking.mlflow import log_preprocessing_artifacts

    monkeypatch.setenv("HOME", str(tmp_path))
    original = comparison_config.model_dump(mode="json")
    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("turboswan.mlops.tracking.mlflow.mlflow.log_dict") as log_dict,
        patch("turboswan.mlops.tracking.mlflow.mlflow.log_artifact") as log_artifact,
        patch("turboswan.mlops.tracking.mlflow.create_preprocessing_artifacts", return_value=[]),
    ):
        log_preprocessing_artifacts(comparison_config, Mock())

    log_artifact.assert_not_called()
    snapshot, artifact_path = log_dict.call_args.args
    assert artifact_path == "metadata/config.json"
    assert snapshot["data"]["source"] == "~"
    assert snapshot["data"]["min_max_file"] == "min_max_files/min-max-values-all.nc"
    assert snapshot["run"]["model_dir"] == "~"
    assert snapshot["mlflow"]["local_tracking_dir"] == "~/mlflow"
    assert snapshot["comparisons"][0]["subdir"] == "."
    assert snapshot["training"] == original["training"]
    assert comparison_config.model_dump(mode="json") == original
    restored = training.TrainConfig.model_validate(snapshot)
    assert restored.data == comparison_config.data
    assert restored.run == comparison_config.run
    assert restored.comparisons == comparison_config.comparisons
    assert restored.mlflow is not None
    assert restored.mlflow.local_tracking_dir is not None
    assert restored.mlflow.local_tracking_dir.expanduser() == comparison_config.mlflow.local_tracking_dir


@pytest.mark.parametrize("mode", ["local", "dnai"])
def test_dataset_sources_record_portable_locations_and_stable_manifests(
    comparison_config, tmp_path: Path, mode: str
) -> None:
    from mlflow.data.pandas_dataset import from_pandas

    from turboswan.mlops.tracking.metadata import (
        build_run_metadata,
        tfrecord_manifest_digest,
    )
    from turboswan.mlops.tracking.mlflow import log_tfrecord_inputs

    if mode == "dnai":
        settings = comparison_config.model_dump(mode="json")
        settings["run"]["mode"] = "dnai"
        settings["mlflow"].pop("local_tracking_dir")
        comparison_config = training.TrainConfig.model_validate(settings)
    files = [str(tmp_path / "train_data" / "example.tfrecords")]
    with (
        patch("pathlib.Path.home", return_value=tmp_path),
        patch("turboswan.mlops.tracking.mlflow.training.list_tfrecords", return_value=files),
        patch(
            "turboswan.mlops.tracking.mlflow.from_pandas",
            wraps=from_pandas,
        ) as from_pandas,
        patch("turboswan.mlops.tracking.mlflow.mlflow.log_input"),
        patch("turboswan.mlops.tracking.mlflow.mlflow.log_table"),
    ):
        metadata = build_run_metadata(comparison_config, experiment_name="turboswan/test")
        log_tfrecord_inputs(comparison_config)

    expected_root = "~"
    assert metadata.params["Data - source"] == expected_root
    assert metadata.params["Run - mode"] == mode
    assert [call.kwargs["source"] for call in from_pandas.call_args_list] == [
        f"{expected_root}/train_data", f"{expected_root}/val_data"
    ]
    assert from_pandas.call_args_list[0].args[0]["tfrecord_file"].tolist() == ["example.tfrecords"]
    assert tfrecord_manifest_digest(files) == tfrecord_manifest_digest(
        ["/another/machine/train_data/example.tfrecords"]
    )
    assert tfrecord_manifest_digest(files) != tfrecord_manifest_digest(["different.tfrecords"])


def test_tfrecord_manifests_persist_selected_file_references(tmp_path: Path) -> None:
    import json

    import mlflow
    from mlflow.tracking import MlflowClient

    from turboswan.mlops.tracking.metadata import recorded_data_source
    from turboswan.mlops.tracking.mlflow import configure_run_tracking, log_tfrecord_inputs

    config = training.TrainConfig.model_validate(
        {
            "data": {"source": str(tmp_path / "data"), "max_tfrecords": 2},
            "run": {"model_dir": str(tmp_path / "models"), "model_version": "test"},
        }
    )
    for split in ("train", "val"):
        directory = training.resolve_split_dir(config.data, split)
        directory.mkdir(parents=True)
        for name in ("c.tfrecords", "a.tfrecords", "b.tfrecords"):
            (directory / name).write_bytes(b"training-data-not-to-upload")
    uri = configure_run_tracking(
        "turboswan/manifest-test", mode="local", local_tracking_dir=tmp_path / "mlflow"
    )
    with mlflow.start_run() as run:
        log_tfrecord_inputs(config)

    client = MlflowClient(uri)
    stored_run = client.get_run(run.info.run_id)
    for split, context in (("train", "training"), ("val", "validation")):
        artifact_file = f"metadata/{split}-tfrecords.json"
        manifest = mlflow.load_table(artifact_file, run_ids=[run.info.run_id])
        dataset = next(
            item.dataset for item in stored_run.inputs.dataset_inputs
            if item.dataset.name == f"TurboSWAN {context} TFRecords"
        )
        assert manifest.columns.tolist() == ["tfrecord_file", "file_path"]
        assert manifest["tfrecord_file"].tolist() == ["a.tfrecords", "b.tfrecords"]
        assert json.loads(dataset.source)["uri"] == recorded_data_source(
            config, training.resolve_split_dir(config.data, split)
        )
        assert dataset.digest
        assert {"path": artifact_file, "type": "table"} in json.loads(
            stored_run.data.tags["mlflow.loggedArtifacts"]
        )
    assert {item.path for item in client.list_artifacts(run.info.run_id, "metadata")} == {
        "metadata/train-tfrecords.json", "metadata/val-tfrecords.json",
    }


def test_recipe_labels_are_applied_to_metrics_and_parameters(tmp_path: Path) -> None:
    from turboswan.mlops.tracking.metadata import build_run_metadata, default_labels
    from turboswan.mlops.tracking.mlflow import MetricsCallback

    config = training.TrainConfig.model_validate(
        {
            "data": {"source": str(tmp_path), "output_variables": ["hs", "new_field"]},
            "run": {"model_dir": str(tmp_path), "model_version": "test"},
            "mlflow": {
                "experiment_name": "turboswan/test",
                "metric_labels": {"loss": "Training masked MSE", "val_loss": "Validation masked MSE"},
                "parameter_labels": {
                    "training.batch_size": "Training - samples per optimisation step",
                    "data.output_variables": "Data - predicted variables",
                },
                "variable_descriptions": {"hs": "significant wave height"},
            },
        }
    )

    assert config.mlflow is not None
    metadata = build_run_metadata(config, experiment_name=config.mlflow.experiment_name)
    assert metadata.params["Training - samples per optimisation step"] == config.training.batch_size
    assert "training.batch_size" not in metadata.params
    assert metadata.params["Data - predicted variables"] == "hs (significant wave height), new_field"
    # Defaults label the shared settings; the recipe only overrides the names it lists.
    assert metadata.params["Run - model version"] == "test"
    assert "Data - random seed" in metadata.params
    assert "run.model_dir" not in metadata.params
    expected = {
        key: label for key, label in default_labels()["parameter_labels"].items()
        if not key.startswith("comparison.")
    }
    expected["training.batch_size"] = "Training - samples per optimisation step"
    assert set(metadata.params) == set(expected.values())
    assert config.model_dump(mode="json")["training"]["lr_decay_rate"] == config.training.lr_decay_rate

    logs = {"loss": 0.4, "val_loss": 0.5, "mean_absolute_error": 0.2}
    with patch("turboswan.mlops.tracking.mlflow.mlflow.log_metrics") as log_metrics:
        MetricsCallback(config.mlflow.metric_labels).on_epoch_end(2, logs)
    log_metrics.assert_called_once_with(
        {"Training masked MSE": 0.4, "Validation masked MSE": 0.5, "mean_absolute_error": 0.2},
        step=3,
    )
    assert logs == {"loss": 0.4, "val_loss": 0.5, "mean_absolute_error": 0.2}


def test_local_notebook_settings_do_not_require_platform_access(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import ast
    import json
    notebook_path = Path(__file__).parents[1] / "notebooks" / "demo_tracking.ipynb"
    notebook = json.loads(notebook_path.read_text())
    # The cells that read the recipe, show its sections and locate the local MLflow folder.
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and any(
            marker in "".join(cell["source"])
            for marker in ("import sys", "RECIPE_FILE =", "show_recipe_section", "record_folder =")
        )
    )
    assert "RECORD_LOCATION" not in code
    assert '"dnai"' not in code

    monkeypatch.chdir(notebook_path.parent)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    data_dir = tmp_path / "turboswan" / "data" / "demo"
    for directory in ("train_data", "val_data", "min_max_files"):
        (data_dir / directory).mkdir(parents=True)
    (data_dir / "min_max_files" / "min-max-values-all.nc").touch()
    (data_dir / "valid-mask.npy").touch()
    namespace = {"Path": Path, "training": training}

    with patch("turboswan.training.check_inputs") as check_inputs, patch("subprocess.Popen") as start_ui:
        exec(compile(ast.parse(code), str(notebook_path), "exec"), namespace)

    start_ui.assert_called_once()
    assert "sqlite:///" + str(tmp_path / "turboswan" / "mlflow" / "mlflow.db") in start_ui.call_args.args[0]
    recipe = namespace["recipe"]
    assert recipe.run.mode == "local"
    assert recipe.mlflow.model_name == "turboswan-hs-demo"
    assert recipe.mlflow.experiment_name == "turboswan/demo-training"
    assert recipe.mlflow.run_name is None
    assert recipe.mlflow.description == "Small demonstration using the TurboSWAN demo recipe."
    shown = capsys.readouterr().out
    for section in ("[data]", "[model]", "[training]", "[run]", "[mlflow]", "[[comparisons]]"):
        assert section in shown
    assert namespace["record_folder"] == (tmp_path / "turboswan" / "mlflow").resolve()
    assert recipe.training.n_epochs == 3
    assert recipe.model.widths == [4, 8, 8, 8, 8, 8]
    assert recipe.data.data_dir == data_dir
    assert recipe._recipe_toml is not None
    check_inputs.assert_not_called()


@pytest.mark.parametrize("labels", [{"loss": "Invalid (label)"}, {"loss": "val_loss"}])
def test_metric_labels_reject_invalid_names_and_collisions(labels) -> None:
    from turboswan.mlops.tracking.mlflow import MetricsCallback

    with patch("turboswan.mlops.tracking.mlflow.mlflow.log_metrics") as log_metrics:
        with pytest.raises(ValueError):
            MetricsCallback(labels).on_epoch_end(0, {"loss": 0.4, "val_loss": 0.5})
    log_metrics.assert_not_called()


def test_training_history_plot_accepts_metrics_without_validation(
    tmp_path: Path,
) -> None:
    from turboswan.mlops.tracking.artifacts import create_training_history_plot

    history = keras.callbacks.History()
    history.history = {"loss": [0.8, 0.4], "mean_absolute_error": [0.6, 0.2]}
    destination = tmp_path / "history.png"

    create_training_history_plot(history, destination)

    assert destination.stat().st_size > 0


def test_training_history_plot_uses_recipe_labels(tmp_path: Path) -> None:
    from turboswan.mlops.tracking.artifacts import create_training_history_plot

    history = keras.callbacks.History()
    history.history = {"loss": [0.8, 0.4], "other_metric": [0.6, 0.2]}
    with patch("turboswan.mlops.tracking.artifacts.plt.close"):
        create_training_history_plot(
            history, tmp_path / "history.png", metric_labels={"loss": "Training masked MSE"}
        )
    figure = plt.gcf()
    try:
        assert [line.get_label() for line in figure.axes[0].lines] == ["Training masked MSE", "other_metric"]
    finally:
        plt.close(figure)


def test_model_signature_comes_from_the_fitted_model() -> None:
    from mlflow.types.schema import TensorSpec

    from turboswan.mlops.tracking.mlflow import build_model_signature

    raster = keras.Input(shape=(4, 3), name="custom_raster")
    boundary = keras.Input(shape=(2,), name="custom_boundary")
    flattened = keras.layers.Flatten()(raster)
    output = keras.layers.Concatenate(name="custom_output")([flattened, boundary])
    model = keras.Model(inputs=[raster, boundary], outputs=output)

    signature = build_model_signature(model)

    assert signature.inputs.input_names() == ["custom_raster", "custom_boundary"]
    input_spec = signature.inputs.inputs[0]
    assert isinstance(input_spec, TensorSpec)
    assert input_spec.shape == (-1, 4, 3)
    assert signature.outputs.input_names() == ["custom_output"]


@pytest.fixture
def comparison_config(tmp_path: Path) -> training.TrainConfig:
    return training.TrainConfig.model_validate(
        {
            "data": {"source": str(tmp_path)},
            "run": {"model_dir": str(tmp_path), "model_version": "test"},
            "mlflow": {
                "experiment_name": "turboswan/comparison-test",
                "local_tracking_dir": str(tmp_path / "mlflow"),
            },
            "comparisons": [
                {
                    "name": "Selected example",
                    "subdir": ".",
                    "plots": ["Error map"],
                }
            ],
        }
    )


def test_local_notebook_recipe_keeps_demo_training_choices() -> None:
    configs = Path(__file__).parents[1] / "configs"
    base = training.load_config(configs / "demo.toml")
    local = training.load_config(configs / "demo_mlflow_local.toml")

    assert base.comparisons == []
    assert base.model_dump(exclude={"comparisons": True, "mlflow": {"experiment_name", "model_name", "description"}}) == local.model_dump(
        exclude={"comparisons": True, "mlflow": {"experiment_name", "model_name", "description"}}
    )
    assert local.mlflow is not None and base.mlflow is not None
    assert local.mlflow.experiment_name != base.mlflow.experiment_name
    assert local.comparisons[0].plots == ["Error map"]
    assert local.comparisons[0].subdir == "val_data"
    assert local.data.data_dir == (Path.home() / "turboswan" / "data" / "demo").resolve()
    assert local.mlflow is not None
    assert local.mlflow.metric_labels["val_loss"] == "Validation masked MSE - normalised wave height"
    assert local.mlflow.variable_descriptions == {}

    from turboswan.mlops.tracking.metadata import build_run_metadata, recorded_data_source

    metadata = build_run_metadata(local, experiment_name=local.mlflow.experiment_name)
    assert metadata.params["Training - samples per optimisation step"] == local.training.batch_size
    assert metadata.params["Data - predicted variables"] == "hs (significant wave height)"
    assert metadata.params["Data - source"] == "~/turboswan/data/demo"
    assert metadata.params["Data - training split"] == "train_data"
    assert metadata.params["Data - validation split"] == "val_data"
    assert metadata.params["Run - mode"] == "local"
    assert metadata.params["Training - learning-rate decay factor"] == local.training.lr_decay_rate
    assert not any(name.startswith(("data.", "training.", "run.", "augmentation.")) for name in metadata.params)
    assert "comparisons" not in metadata.params

    settings = base.model_dump(mode="json")
    settings["run"]["mode"] = "dnai"
    settings["mlflow"].pop("local_tracking_dir", None)
    platform = training.TrainConfig.model_validate(settings)
    metadata = build_run_metadata(platform, experiment_name=platform.mlflow.experiment_name)
    assert metadata.params["Run - mode"] == "dnai"
    assert metadata.params["Data - source"] == recorded_data_source(platform, platform.data.data_dir)


def _recipe(tmp_path: Path, **sections: dict) -> dict:
    return {
        "data": {"source": str(tmp_path)},
        "run": {"model_dir": str(tmp_path), "model_version": "test"},
        **sections,
    }


def test_run_mode_rejects_local_tracking_dir_in_dnai_mode(tmp_path: Path) -> None:
    raw = _recipe(
        tmp_path,
        run={"model_dir": str(tmp_path), "model_version": "test", "mode": "dnai"},
        mlflow={"experiment_name": "x", "local_tracking_dir": "/tmp/mlflow"},
    )
    with pytest.raises(ValueError, match="local_tracking_dir only applies"):
        training.TrainConfig.model_validate(raw)


def test_check_inputs_raises_unless_the_data_folder_exists(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    config = training.TrainConfig.model_validate(_recipe(tmp_path, data={"source": str(data_dir)}))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        training.check_inputs(config.data)
    data_dir.mkdir()
    assert training.check_inputs(config.data) == data_dir.resolve()


def test_recipe_section_shows_the_recipe_as_written(tmp_path: Path, capsys) -> None:
    recipe_file = tmp_path / "recipe.toml"
    recipe_file.write_text(
        f"""[data]
# where the files are
source = "{tmp_path}"  # a folder

[run]
model_dir = "{tmp_path}"
model_version = "test"
# describes the table below, not [run]
[mlflow]
experiment_name = "turboswan/test"

[mlflow.metric_labels]
loss = "Training loss"

[[comparisons]]
name = "first"
subdir = "."
plots = ["Error map"]

[[comparisons]]
name = "second"
subdir = "."
plots = ["Error map"]
"""
    )
    recipe = training.load_config(recipe_file)

    assert training.recipe_section(recipe, "data") == (
        f'[data]\n# where the files are\nsource = "{tmp_path}"  # a folder'
    )
    assert training.recipe_section(recipe, "run").endswith('model_version = "test"')
    assert training.recipe_section(recipe, "mlflow") == (
        '[mlflow]\nexperiment_name = "turboswan/test"\n\n'
        "# also set in this recipe: [mlflow.metric_labels]"
    )
    assert training.recipe_section(recipe, "mlflow.metric_labels") == (
        '[mlflow.metric_labels]\nloss = "Training loss"'
    )
    both = training.recipe_section(recipe, "comparisons")
    assert both.count("[[comparisons]]") == 2 and 'name = "second"' in both
    assert "defaults apply" in training.recipe_section(recipe, "augmentation")
    training.show_recipe_section(recipe, "run")
    assert capsys.readouterr().out.startswith("[run]")
    with pytest.raises(ValueError, match="Unknown section"):
        training.recipe_section(recipe, "nope")


def test_recipe_section_needs_a_recipe_file(comparison_config) -> None:
    with pytest.raises(ValueError, match="not loaded from a recipe file"):
        training.recipe_section(comparison_config, "data")


def test_plain_training_does_not_silently_ignore_plots(comparison_config) -> None:
    with patch("turboswan.training.check_inputs") as check_inputs:
        with pytest.raises(ValueError, match="tracked run"):
            training.run_training(comparison_config)
    check_inputs.assert_not_called()


@pytest.mark.parametrize("invalid_input", ["no_hs", "missing_directory", "no_shards"])
def test_tracked_training_rejects_missing_inputs_before_training(
    comparison_config, invalid_input
) -> None:
    from turboswan.training_mlflow import track_training_run

    if invalid_input == "no_hs":
        comparison_config.data.output_variables = ["tp"]
        error, message = ValueError, "requires 'hs'"
    elif invalid_input == "missing_directory":
        comparison_config.comparisons[0].subdir = "missing"
        error, message = FileNotFoundError, "directory not found"
    else:
        error, message = FileNotFoundError, "No .tfrecords"
    with patch("turboswan.training_mlflow.training.run_training") as run_training:
        with pytest.raises(error, match=message):
            track_training_run(comparison_config)
    run_training.assert_not_called()


def test_preprocessing_artifacts_use_recipe_variable_descriptions(
    comparison_config, tmp_path: Path
) -> None:
    import csv

    from turboswan.mlops.tracking.artifacts import create_preprocessing_artifacts

    comparison_config.mlflow.variable_descriptions = {"hs": "recipe wave height"}
    variables = (
        comparison_config.data.input_variables
        + comparison_config.data.output_variables
        + comparison_config.data.boundary_variables
    )
    ranges = xr.Dataset({name: (("index",), [0.0, 1.0]) for name in variables})
    history = keras.callbacks.History()
    history.history = {"loss": [0.4], "val_loss": [0.5]}
    with patch(
        "turboswan.mlops.tracking.artifacts.training.load_normalization",
        return_value=(ranges, np.ones((2, 2))),
    ):
        create_preprocessing_artifacts(comparison_config, history, tmp_path)

    with (tmp_path / "normalization-ranges.csv").open() as ranges_file:
        rows = {row["variable"]: row for row in csv.DictReader(ranges_file)}
    assert rows["hs"]["meaning"] == "recipe wave height"
    assert rows["xwnd"]["meaning"] == "east-west wind component"


@pytest.mark.parametrize("output_variables", [["hs"], ["tp", "hs"]])
def test_error_map_uses_selected_channel_metres_and_water_mask(
    comparison_config, output_variables
) -> None:
    comparison_config.data.output_variables = output_variables
    comparison_config.mlflow.parameter_labels = {
        f"comparison.{field}": f"Comparison {{index}} - {label}"
        for field, label in (
            ("name", "name"), ("subdir", "data folder"),
            ("sample_file", "source file"), ("reference", "reference"),
        )
    }
    inputs = (tf.zeros((2, 2, 3)), tf.zeros((106, 3)))
    hs = np.array([[0.5, 0.5], [0.0, 0.25]], dtype=np.float32)
    mask = np.array([[1, 0], [1, 1]], dtype=np.float32)
    channels = [hs if name == "hs" else np.ones_like(hs) for name in output_variables]
    targets = tf.constant(np.stack([*channels, mask], axis=-1))
    dataset = tf.data.Dataset.from_tensors((inputs, targets))
    model = Mock()
    model.predict.return_value = np.full((1, 2, 2, len(output_variables)), 0.25)
    ranges = xr.Dataset({"hs": (("index",), [2.0, 6.0])})

    with (
        patch(
            "turboswan.mlops.tracking.comparisons.training.load_normalization",
            return_value=(ranges, mask),
        ),
        patch(
            "turboswan.mlops.tracking.comparisons.training.list_tfrecords",
            return_value=["first.tfrecords", "second.tfrecords"],
        ),
        patch(
            "turboswan.mlops.tracking.comparisons.create_datasets.get_dataset",
            return_value=dataset,
        ) as get_dataset,
        patch("turboswan.mlops.tracking.comparisons.mlflow.log_params") as log_params,
        patch("turboswan.mlops.tracking.comparisons.mlflow.log_figure") as log_figure,
    ):
        log_comparisons(comparison_config, model)

    assert get_dataset.call_args.args[0] == ["first.tfrecords"]
    assert get_dataset.call_args.kwargs["output_variables"] == output_variables
    assert log_params.call_args.args[0] == {
        "Comparison 1 - name": comparison_config.comparisons[0].name,
        "Comparison 1 - data folder": ".",
        "Comparison 1 - source file": "first.tfrecords",
        "Comparison 1 - reference": "SWAN hs TFRecord target",
    }
    figure, artifact_path = log_figure.call_args.args
    assert artifact_path == "comparisons/01/error-map-hs.png"
    image = figure.axes[0].images[0]
    np.testing.assert_allclose(
        image.get_array().filled(np.nan), [[1.0, np.nan], [-1.0, 0.0]], equal_nan=True
    )
    assert image.get_clim() == (-1.0, 1.0)
    assert figure.axes[1].get_ylabel() == "wave-height difference (m)"
    assert figure.number not in plt.get_fignums()
    assert model.predict.call_args.kwargs == {"verbose": 0}


def test_error_map_reads_first_selected_tfrecord_example(
    comparison_config, tmp_path
) -> None:
    ranges = xr.Dataset(
        {
            name: (("index",), [2.0, 6.0])
            for name in ("xwnd", "ywnd", "wl", "hs", "hs_bnd", "tp_bnd", "theta0_bnd")
        }
    )
    for filename, wave_heights in [("b.tfrecords", [6.0]), ("a.tfrecords", [4.0, 6.0])]:
        writer = tf.io.TFRecordWriter(str(tmp_path / filename))
        with writer:
            for wave_height in wave_heights:
                features = {
                    name: tf.train.Feature(int64_list=tf.train.Int64List(value=[2]))
                    for name in ("height", "width", "depth")
                }
                tensors = {
                    name: tf.zeros((2, 2, 4)) for name in ("xwnd", "ywnd", "wl")
                }
                tensors.update(
                    {name: tf.zeros((106,)) for name in ("hs_bnd", "tp_bnd", "theta0_bnd")}
                )
                tensors["hs"] = tf.fill((2, 2), wave_height)
                features.update(
                    {
                        name: tf.train.Feature(
                            bytes_list=tf.train.BytesList(
                                value=[tf.io.serialize_tensor(tensor).numpy()]
                            )
                        )
                        for name, tensor in tensors.items()
                    }
                )
                writer.write(
                    tf.train.Example(
                        features=tf.train.Features(feature=features)
                    ).SerializeToString()
                )
    model = Mock()
    model.predict.return_value = np.full((1, 2, 2, 1), 0.25)
    with (
        patch(
            "turboswan.mlops.tracking.comparisons.training.load_normalization",
            return_value=(ranges, np.array([[1, 0], [1, 1]])),
        ),
        patch("turboswan.mlops.tracking.comparisons.training.RASTER_HEIGHT", 2),
        patch("turboswan.mlops.tracking.comparisons.training.RASTER_WIDTH", 2),
        patch("turboswan.mlops.tracking.comparisons.mlflow.log_params") as log_params,
        patch("turboswan.mlops.tracking.comparisons.mlflow.log_figure") as log_figure,
    ):
        log_comparisons(comparison_config, model)
    assert log_params.call_args.args[0]["Comparison 1 - source file"] == "a.tfrecords"
    figure = log_figure.call_args.args[0]
    np.testing.assert_allclose(
        figure.axes[0].images[0].get_array().filled(np.nan),
        [[1.0, np.nan], [1.0, 1.0]],
        equal_nan=True,
    )
    assert model.predict.call_args.args[0][0].shape == (1, 2, 2, 3)


@pytest.mark.parametrize("model_name", [None, "coastal-baseline"])
def test_tracked_training_logs_maps_in_the_active_run(comparison_config, tmp_path, model_name):
    from types import SimpleNamespace

    import mlflow
    from mlflow.tracking import MlflowClient

    from turboswan.training_mlflow import track_training_run

    if model_name is not None:
        comparison_config.mlflow = training.MLflowConfig.model_validate(
            {**comparison_config.mlflow.model_dump(), "model_name": model_name}
        )
    (tmp_path / "selected.tfrecords").touch()
    history = SimpleNamespace(history={"loss": [0.4]})
    model = Mock()
    comparison_config.comparisons.append(
        comparison_config.comparisons[0].model_copy(update={"name": "Second comparison"})
    )
    mask = np.array([[1, 0], [1, 1]], dtype=np.float32)
    targets = tf.constant(np.stack([np.full((2, 2), 0.5), mask], axis=-1))
    dataset = tf.data.Dataset.from_tensors(
        ((tf.zeros((2, 2, 3)), tf.zeros((106, 3))), targets)
    )
    model.predict.return_value = np.full((1, 2, 2, 1), 0.25)
    with (
        patch(
            "turboswan.training_mlflow.training.run_training",
            return_value=(history, model),
        ) as run_training,
        patch(
            "turboswan.mlops.tracking.comparisons.training.load_normalization",
            return_value=(xr.Dataset({"hs": (("index",), [2.0, 6.0])}), mask),
        ),
        patch(
            "turboswan.mlops.tracking.comparisons.create_datasets.get_dataset",
            return_value=dataset,
        ),
        patch("turboswan.training_mlflow.log_preprocessing_artifacts"),
        patch("turboswan.training_mlflow.build_model_signature"),
        patch(
            "turboswan.training_mlflow.mlflow.keras.log_model",
            return_value=SimpleNamespace(model_uri="test-model-uri", model_id="test-model-id"),
        ) as log_model,
        patch("turboswan.training_mlflow.log_tfrecord_inputs"),
    ):
        result = track_training_run(comparison_config)

    assert run_training.call_args.kwargs["return_model"] is True
    assert log_model.call_args.kwargs["name"] == (model_name or "model")
    assert log_model.call_args.kwargs["registered_model_name"] is None
    assert "pip_requirements" not in log_model.call_args.kwargs
    assert result.model_uri == "test-model-uri"
    assert result.model_id == "test-model-id"
    assert result.history is history
    client = MlflowClient(result.tracking_uri)
    run = client.get_run(result.run_id)
    assert run.info.status == "FINISHED"
    assert run.data.params["Comparison 1 - source file"] == "selected.tfrecords"
    assert run.data.params["Comparison 2 - name"] == "Second comparison"
    assert mlflow.active_run() is None
    for index in (1, 2):
        path = f"comparisons/{index:02d}/error-map-hs.png"
        artifact = Path(client.download_artifacts(result.run_id, path))
        assert artifact.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


@pytest.mark.parametrize("failing_report", ["log_comparisons", "log_preprocessing_artifacts"])
def test_report_failure_preserves_logged_model(comparison_config, tmp_path, failing_report):
    import mlflow
    import mlflow.keras as mlflow_keras
    from mlflow.tracking import MlflowClient

    from turboswan.training_mlflow import track_training_run

    (tmp_path / "comparison.tfrecords").touch()
    for split in ("train", "val"):
        directory = training.resolve_split_dir(comparison_config.data, split)
        directory.mkdir()
        (directory / "selected.tfrecords").touch()
    history = keras.callbacks.History()
    history.history = {"loss": [0.4]}
    inputs = keras.Input(shape=(2,))
    model = keras.Model(inputs, keras.layers.Dense(1)(inputs))
    sample = np.ones((1, 2), dtype=np.float32)
    expected = model(sample).numpy()
    saved: list[tuple[str, str]] = []
    log_model = mlflow_keras.log_model

    def capture_model(*args, **kwargs):
        info = log_model(*args, **kwargs)
        run = mlflow.active_run()
        assert run is not None
        saved.append((run.info.run_id, info.model_uri))
        return info

    with (
        patch(
            "turboswan.training_mlflow.training.run_training",
            return_value=(history, model),
        ),
        patch("turboswan.training_mlflow.log_comparisons") as comparisons,
        patch("turboswan.training_mlflow.log_preprocessing_artifacts") as preprocessing,
        patch("turboswan.training_mlflow.mlflow_keras.log_model", side_effect=capture_model),
    ):
        reports = {"log_comparisons": comparisons, "log_preprocessing_artifacts": preprocessing}
        reports[failing_report].side_effect = RuntimeError("report failed")
        with pytest.raises(RuntimeError, match="report failed"):
            track_training_run(comparison_config)

    assert len(saved) == 1
    run_id, model_uri = saved[0]
    client = MlflowClient()
    assert client.get_run(run_id).info.status == "FAILED"
    assert mlflow.active_run() is None
    restored = mlflow_keras.load_model(model_uri)
    assert isinstance(restored, keras.Model)
    np.testing.assert_allclose(restored(sample).numpy(), expected)
    for split in ("train", "val"):
        manifest = mlflow.load_table(
            f"metadata/{split}-tfrecords.json", run_ids=[run_id]
        )
        assert manifest["tfrecord_file"].tolist() == ["selected.tfrecords"]
