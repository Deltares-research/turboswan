"""Run a whole sweep in one call that returns once every trial is scored."""

import logging

import mlflow
from mlflow.tracking import MlflowClient

from turboswan.mlops.sweeps.blocking.executors import BlockingExecutor, SequentialExecutor
from turboswan.mlops.sweeps.configuration import (
	Categorical,
	Distribution,
	StrategyConfig,
	SweepConfig,
)
from turboswan.mlops.sweeps.parent_run import (
	configure_tracking,
	finalize_sweep,
	log_sweep_inputs,
	parent_tags,
	score,
)
from turboswan.mlops.sweeps.search import build_search_strategy
from turboswan.mlops.sweeps.trial import run_trial

log = logging.getLogger(__name__)


def run_sweep(
	sweep: SweepConfig,
	strategy: StrategyConfig,
	search_space: dict[str, Categorical | Distribution],
	base_config: dict,
	*,
	sweep_id: str,
	executor: BlockingExecutor | None = None,
	resume_study: str | None = None,
	recipe_toml: bytes | None = None,
) -> dict:
	"""Execute a configured sweep under one MLflow parent run and return every trial plus the winner."""
	if strategy.pruner is not None:
		raise ValueError("pruning needs the stepped sweep; remove [strategy.pruner] or use step_sweep")
	# initialize tracking, executor and search strategy
	mlflow_config, tracking_uri = configure_tracking(base_config)
	client = MlflowClient(tracking_uri)
	executor = SequentialExecutor() if executor is None else executor
	search = build_search_strategy(
		sweep, strategy, search_space, base_config, resume_study=resume_study,
	)
	# start the parent MLflow run
	with mlflow.start_run(run_name=f"sweep-{sweep_id}") as parent:
		# log search space and sweep/search settings
		log_sweep_inputs(
			client, parent.info.run_id, sweep, strategy, search_space,
			parent_tags(mlflow_config.tags, sweep_id, search),
		)
		results = []
		submitted_overrides = {}
		first_index = search.index_offset
		next_index = first_index
		in_flight = 0
		try: # try/finally to ensure no jobs are left running in case of error
			# keep looping until all jobs have been submitted and none are running
			while next_index < first_index + strategy.max_configs or in_flight:
				# fill up the available slots with new trials
				while next_index < first_index + strategy.max_configs and in_flight < strategy.max_concurrent:
					overrides = {
						**search.ask(next_index),
						"run.model_version": f"{base_config['run']['model_version']}-cfg{next_index:02d}",
					}
					submitted_overrides[next_index] = overrides
					executor.submit(
						next_index,
						run_trial,
						base_config=base_config,
						overrides=overrides,
						config_index=next_index,
						sweep_id=sweep_id,
						parent_run_id=parent.info.run_id,
						recipe_toml=recipe_toml,
					)
					next_index += 1
					in_flight += 1
				config_index, outcome = executor.wait_next()
				in_flight -= 1
				result, value = score(
					outcome, submitted_overrides.pop(config_index), config_index, sweep.selection_metric
				)
				results.append(result)
				search.tell(config_index, value)
				log.info(
					"sweep progress: %d/%d done, %d in flight, config_index=%d %s=%s",
					len(results), strategy.max_configs, in_flight, config_index,
					sweep.selection_metric, value,
				)
		finally:
			executor.cancel_all()
		results.sort(key=lambda result: result["config_index"])
		# determine best trial, register model if best one across the whole study
		winner = finalize_sweep(
			client, parent.info.run_id, results, sweep,
			search.prior_best, mlflow_config.registered_model_name,
		)
	return {"parent_run_id": parent.info.run_id, "results": results, "winner": winner}


__all__ = ["run_sweep"]
