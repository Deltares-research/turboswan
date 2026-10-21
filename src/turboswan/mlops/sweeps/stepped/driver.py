"""Advance a sweep by one poll per call, persisting its state between calls."""

import logging
import pathlib

from mlflow.tracking import MlflowClient

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
from turboswan.mlops.sweeps.search import SearchStrategy, build_search_strategy
from turboswan.mlops.sweeps.stepped.executors import SteppedExecutor
from turboswan.mlops.sweeps.stepped.state import PendingTrial, SweepState
from turboswan.mlops.sweeps.trial import failed_trial

log = logging.getLogger(__name__)


def _trial_runs(client: MlflowClient, experiment_id: str, parent_run_id: str) -> dict[int, str]:
	"""Map config_index to run id for the child runs the jobs have opened so far."""
	runs = client.search_runs(
		[experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent_run_id}'"
	)
	return {int(run.data.tags["config_index"]): run.info.run_id for run in runs}


def _epoch_history(client: MlflowClient, run_id: str, metric: str) -> dict[int, float]:
	"""Return a run's per-epoch values of ``metric`` by step."""
	return {m.step: m.value for m in client.get_metric_history(run_id, metric)}


def _record_submit_failure(
	search: SearchStrategy,
	state: SweepState,
	state_path: pathlib.Path,
	config_index: int,
	overrides: dict,
	error: Exception,
) -> None:
	"""Close a trial that was asked but never submitted, so a rerun moves on to the next index."""
	search.tell(config_index, None)
	state.results.append(failed_trial(overrides, config_index, f"submit failed: {error!r}"))
	state.next_index += 1
	state.save(state_path)


def step_sweep(
	sweep: SweepConfig,
	strategy: StrategyConfig,
	search_space: dict[str, Categorical | Distribution],
	base_config: dict,
	*,
	sweep_id: str,
	state_path: pathlib.Path,
	executor: SteppedExecutor,
	resume_study: str | None = None,
	recipe_toml: bytes | None = None,
) -> bool:
	"""Advance a sweep by one poll; return True once every config is scored."""
	state = SweepState.model_validate_json(state_path.read_text()) if state_path.exists() else None
	if state is not None and state.finished:
		return True

	# 1. create or load
	mlflow_config = configure_tracking(base_config)
	client = MlflowClient()
	experiment = client.get_experiment_by_name(mlflow_config.experiment_name)
	if state is None:
		search = build_search_strategy(
			sweep, strategy, search_space, base_config, resume_study=resume_study,
		)
		parent = client.create_run(experiment.experiment_id, run_name=f"sweep-{sweep_id}")
		log_sweep_inputs(
			client, parent.info.run_id, sweep, strategy, search_space,
			parent_tags(mlflow_config.tags, sweep_id, search),
		)
		study_name = getattr(search, "mlflow_tags", {}).get("optuna_study_name")
		first_index = getattr(search, "index_offset", 0)
		state = SweepState(
			parent_run_id=parent.info.run_id,
			study_name=study_name,
			prior_best=getattr(search, "prior_best", None),
			first_index=first_index,
			next_index=first_index,
		)
	else:
		# pin the study, or every step would start a new one
		search = build_search_strategy(
			sweep, strategy, search_space, base_config, resume_study=state.study_name,
		)
	executor.connect()
	runs = _trial_runs(client, experiment.experiment_id, state.parent_run_id)

	# 2. update; save per trial so a crash never tells optuna twice
	jobs = {config_index: trial.job_id for config_index, trial in state.pending.items()}
	for config_index, outcome in executor.poll(jobs).items():
		result, value = score(
			outcome, state.pending[config_index].overrides, config_index, sweep.selection_metric
		)
		# update search strategy with observed metric
		search.tell(config_index, value)
		if "error" in outcome and config_index in runs:
			# a killed job never closes its own run
			client.set_terminated(runs[config_index], "FAILED")
		# move job from pending to completed
		state.record(config_index, result)
		state.save(state_path)

	# 3. prune; order is cancel, tell, mlflow, record
	if strategy.pruner is not None:
		# trials log epochs under the recipe's [mlflow.metric_labels], not the Keras name
		logged_metric = mlflow_config.metric_labels.get(sweep.selection_metric, sweep.selection_metric)
		for config_index, trial in list(state.pending.items()):
			# no run yet: the job is still queued
			if config_index not in runs:
				continue
			run_id = runs[config_index]
			history = _epoch_history(client, run_id, logged_metric)
			pruned_at = search.report(config_index, history) if history else None
			if pruned_at is None:
				continue
			# a failed cancel raises with optuna and the state untouched; the next step retries
			executor.cancel(trial.job_id)
			search.tell_pruned(config_index)
			client.set_tag(run_id, "pruned_at_step", str(pruned_at))
			client.set_terminated(run_id, "KILLED")
			state.record(config_index, {
				"config_index": config_index,
				"overrides": trial.overrides,
				"run_id": run_id,
				"pruned_at_step": pruned_at,
			})
			state.save(state_path)

	# 4. submit; save per trial so a crash never orphans an asked trial
	last_index = state.first_index + strategy.max_configs
	while state.next_index < last_index and len(state.pending) < strategy.max_concurrent:
		config_index = state.next_index
		overrides = {
			**search.ask(config_index),
			"run.model_version": f"{base_config['run']['model_version']}-cfg{config_index:02d}",
		}
		try:
			job_id = executor.submit(
				config_index,
				base_config=base_config,
				overrides=overrides,
				sweep_id=sweep_id,
				parent_run_id=state.parent_run_id,
				# str, since executors pass the arguments as JSON
				recipe_toml=None if recipe_toml is None else recipe_toml.decode("utf-8"),
			)
		except Exception as error:
			# retrying is the executor's job; fail loudly, but leave optuna and the state consistent
			_record_submit_failure(search, state, state_path, config_index, overrides, error)
			raise
		state.pending[config_index] = PendingTrial(job_id=job_id, overrides=overrides)
		state.next_index += 1
		state.save(state_path)
	log.info(
		"sweep step: %d/%d done, %d in flight",
		len(state.results), strategy.max_configs, len(state.pending),
	)

	# 5. save, and 6. stop once every config is scored
	state.finished = state.next_index == last_index and not state.pending
	if state.finished:
		state.results.sort(key=lambda result: result["config_index"])
		try:
			finalize_sweep(
				client, state.parent_run_id, state.results, sweep,
				state.prior_best, mlflow_config.registered_model_name,
			)
		except Exception:
			# e.g. every config failed; don't leave the parent run RUNNING
			client.set_terminated(state.parent_run_id, "FAILED")
			raise
		# set mlflow run status to finished
		client.set_terminated(state.parent_run_id)
	state.save(state_path)
	return state.finished


def cancel_sweep(state_path: pathlib.Path, executor: SteppedExecutor) -> int:
	"""Cancel every pending job; rerunning the sweep then records them as failed."""
	if not state_path.exists():
		return 0
	state = SweepState.model_validate_json(state_path.read_text())
	for trial in state.pending.values():
		executor.cancel(trial.job_id)
	return len(state.pending)


__all__ = ["cancel_sweep", "step_sweep"]
