"""Shared fixtures for the opt-in end-to-end tests."""

from pathlib import Path

import pytest

DEMO_CONFIG = Path(__file__).parents[1] / "configs" / "demo.toml"


@pytest.fixture
def e2e_data_dir() -> Path:
    """Locate the manually prepared demo data named by the demo recipe, or skip the test."""
    from turboswan import training

    data_dir = training.load_config(DEMO_CONFIG).data.data_dir
    required = [
        data_dir / "min_max_files" / "min-max-values-all.nc",
        data_dir / "valid-mask.npy",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    missing += [
        f"{data_dir / split}/*.tfrecords"
        for split in ("train_data", "val_data")
        if not any((data_dir / split).glob("*.tfrecords"))
    ]
    if missing:
        pytest.skip(
            f"e2e data not found in {data_dir}; missing: {', '.join(missing)}. "
            "Prepare the folder named by data.source in configs/demo.toml."
        )
    return data_dir.resolve()


@pytest.fixture
def e2e_config_dict(e2e_data_dir: Path, tmp_path: Path) -> dict:
    """Demo recipe pointed at the local data, with every output under tmp_path."""
    from turboswan import training

    return training.apply_overrides(
        training.load_raw_config(DEMO_CONFIG),
        {
            "data.source": str(e2e_data_dir),
            "run.model_dir": str(tmp_path / "models"),
            "mlflow.local_tracking_dir": str(tmp_path / "mlflow"),
            "mlflow.experiment_name": f"turboswan/e2e-{tmp_path.name}",
        },
    )


@pytest.fixture
def e2e_config(e2e_config_dict: dict):
    """Validated form of ``e2e_raw_config``."""
    from turboswan import training

    return training.TrainConfig.model_validate(e2e_config_dict)
