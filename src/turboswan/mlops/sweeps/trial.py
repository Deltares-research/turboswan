"""Train and track one sweep trial as a child of the sweep's parent run; runs on the worker."""

import json
import logging
import pathlib
import sys

from turboswan import training
from turboswan.training_mlflow import track_training_run

log = logging.getLogger(__name__)


def run_trial(
	base_config: dict,
	overrides: dict,
	config_index: int,
	*,
	sweep_id: str,
	parent_run_id: str,
	recipe_toml: bytes | str | None = None,
) -> dict:
	"""Train and track one override set as a child of the sweep's parent run.

	``recipe_toml`` may be a str because stepped executors pass the arguments as JSON.
	"""
	logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
	log.info("config_index=%d overrides=%s", config_index, overrides)
	config = training.TrainConfig.model_validate(training.apply_overrides(base_config, overrides))
	config._recipe_toml = recipe_toml.encode("utf-8") if isinstance(recipe_toml, str) else recipe_toml
	if config.mlflow is None:
		raise ValueError("the sweep's base config needs an [mlflow] section")
	config.mlflow = config.mlflow.model_copy(
		update={
			"run_name": f"{sweep_id}-cfg{config_index:02d}",
			"registered_model_name": None,
			"tags": {
				**config.mlflow.tags,
				"run_type": "sweep_trial",
				"sweep_id": sweep_id,
				"config_index": str(config_index),
				"mlflow.parentRunId": parent_run_id,
			},
		}
	)
	tracked = track_training_run(config, nested=True)
	metrics = {name: float(values[-1]) for name, values in tracked.history.history.items()}
	log.info("config_index=%d metrics=%s", config_index, metrics)
	return {
		"config_index": config_index,
		"overrides": overrides,
		"run_id": tracked.run_id,
		**metrics,
	}


def failed_trial(overrides: dict, config_index: int, error: str) -> dict:
	"""Return a result record for a failed trial without ending the sweep."""
	return {"config_index": config_index, "overrides": overrides, "error": error}


def _run_job(args_path: pathlib.Path) -> None:
	"""Run one trial from a JSON args file and write its result next to it."""
	kwargs = json.loads(args_path.read_text())
	try:
		result = run_trial(**kwargs)
	except Exception as error:
		log.error("config_index=%d failed: %r", kwargs["config_index"], error, exc_info=True)
		result = {"error": repr(error)}
	out = args_path.with_name(f"{kwargs['config_index']}.result.json")
	tmp = out.with_suffix(".tmp")
	tmp.write_text(json.dumps(result))
	tmp.replace(out)


if __name__ == "__main__":
	_run_job(pathlib.Path(sys.argv[1]))

__all__ = ["failed_trial", "run_trial"]
