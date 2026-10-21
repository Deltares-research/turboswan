"""The sweep's MLflow parent run: tracking setup, trial scoring, and winner selection."""

import logging
import math

from deltares_ai_platform_sdk.mlflow_utils import register_model_from_run
from mlflow.entities import Param, RunTag
from mlflow.tracking import MlflowClient

from turboswan import training
from turboswan.mlops.sweeps.configuration import (
	Categorical,
	Distribution,
	StrategyConfig,
	SweepConfig,
)
from turboswan.mlops.sweeps.search import SearchStrategy
from turboswan.mlops.sweeps.trial import failed_trial
from turboswan.mlops.tracking.mlflow import configure_run_tracking

log = logging.getLogger(__name__)


def selection_value(result: dict, selection_metric: str) -> float | None:
	"""Return a trial's selection score, or ``None`` for a failed trial or a missing/NaN metric."""
	if "error" in result:
		return None
	if selection_metric not in result:
		available = sorted(set(result) - {"config_index", "overrides", "run_id"})
		log.warning(
			"selection_metric %r missing from config %d; available: %s",
			selection_metric, result["config_index"], available,
		)
		return None
	value = float(result[selection_metric])
	if math.isnan(value):
		log.warning("selection_metric %r is NaN for config %d", selection_metric, result["config_index"])
		return None
	return value


def score(outcome: dict, overrides: dict, config_index: int, selection_metric: str) -> tuple[dict, float | None]:
	"""Turn an executor outcome into a result record and its selection value."""
	result = failed_trial(overrides, config_index, outcome["error"]) if "error" in outcome else outcome
	value = selection_value(result, selection_metric)
	if value is None and "error" not in result:
		result = failed_trial(overrides, config_index, f"no usable {selection_metric!r}")
	return result, value


def configure_tracking(base_config: dict) -> training.MLflowConfig:
	"""Select the sweep's MLflow tracking and return the base config's [mlflow] section."""
	validated = training.TrainConfig.model_validate(base_config)
	mlflow_config = validated.mlflow
	if mlflow_config is None:
		raise ValueError("the sweep's base config needs an [mlflow] section")
	configure_run_tracking(
		mlflow_config.experiment_name,
		mode=validated.run.mode,
		local_tracking_dir=mlflow_config.local_tracking_dir,
	)
	return mlflow_config


def parent_tags(config_tags: dict, sweep_id: str, search: SearchStrategy) -> dict:
	return {
		**config_tags,
		"project": "turboswan",
		"run_type": "sweep",
		"sweep_id": sweep_id,
		**getattr(search, "mlflow_tags", {}),
	}


def log_sweep_inputs(
	client: MlflowClient,
	run_id: str,
	sweep: SweepConfig,
	strategy: StrategyConfig,
	search_space: dict[str, Categorical | Distribution],
	tags: dict,
) -> None:
	params = {
		**{f"sweep.{key}": value for key, value in sweep.model_dump().items()},
		**{f"strategy.{key}": value for key, value in strategy.model_dump().items()},
	}
	client.log_batch(
		run_id,
		params=[Param(key, str(value)) for key, value in params.items()],
		tags=[RunTag(key, str(value)) for key, value in tags.items()],
	)
	client.log_dict(run_id, {key: spec.model_dump() for key, spec in search_space.items()}, "search_space.json")


def finalize_sweep(
	client: MlflowClient,
	run_id: str,
	results: list[dict],
	sweep: SweepConfig,
	prior_best: float | None,
	registered_model_name: str | None,
) -> dict:
	"""Pick the winner, log it on the parent run, and register it if it beats the study's prior best."""
	metric = sweep.selection_metric
	finished = [result for result in results if "error" not in result and "pruned_at_step" not in result]
	n_pruned = sum("pruned_at_step" in result for result in results)
	n_failed = sum("error" in result for result in results)
	if not finished:
		raise RuntimeError(f"no config in the sweep finished; {n_failed} failed, {n_pruned} pruned")
	if n_failed or n_pruned:
		log.warning("%d of %d configs failed, %d pruned", n_failed, len(results), n_pruned)
	best = max if sweep.direction == "maximize" else min
	winner = best(finished, key=lambda result: result[metric])
	client.log_metric(run_id, f"best_{metric}", winner[metric])
	client.set_tag(run_id, "winner_config_index", str(winner["config_index"]))
	client.set_tag(run_id, "winner_run_id", winner["run_id"])
	beats_prior = prior_best is None or (
		winner[metric] > prior_best if sweep.direction == "maximize" else winner[metric] < prior_best
	)
	if prior_best is not None:
		client.set_tag(run_id, "study_best_before", str(prior_best))
		client.set_tag(run_id, "winner_beats_prior_best", str(beats_prior).lower())
	if registered_model_name and beats_prior:
		version = register_model_from_run(client, registered_model_name, winner["run_id"])
		client.set_tag(run_id, "winner_model_version", version)
	elif registered_model_name:
		log.info(
			"winner %s=%s did not beat the study's earlier best %s; skipping registration",
			metric, winner[metric], prior_best,
		)
	log.info("sweep done: winner config_index=%d %s=%.4f", winner["config_index"], metric, winner[metric])
	return winner


__all__ = [
	"configure_tracking",
	"finalize_sweep",
	"log_sweep_inputs",
	"parent_tags",
	"score",
	"selection_value",
]
