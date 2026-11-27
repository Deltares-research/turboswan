"""Algorithms that propose the next configured parameter combination.

Random and Optuna proposals decide which values to try only within the
researcher-defined search space. They do not establish scientific validity or
acceptance criteria.
"""

from __future__ import annotations

import logging
import random
from typing import Protocol, runtime_checkable

from deltares_ai_platform_sdk import optuna_utils

from turboswan.mlops.sweeps.configuration import (
    Categorical,
    Distribution,
    PrunerConfig,
    SamplerConfig,
    StrategyConfig,
    SweepConfig,
    sample_one,
)
from turboswan.training import read_dotted

log = logging.getLogger(__name__)


@runtime_checkable
class SearchStrategy(Protocol):
    """One configuration out per ``ask`` and one score back per ``tell``."""

    # first config_index; non-zero when resuming a study
    index_offset: int
    # best value among a resumed study's completed trials, or None
    prior_best: float | None
    # extra tags for the sweep's parent run
    mlflow_tags: dict[str, str]

    def ask(self, config_index: int) -> dict:
        """Return dotted-key parameter overrides for the next trial."""
        ...

    def tell(self, config_index: int, value: float | None) -> None:
        """Report a trial selection metric, or ``None`` after a failure."""
        ...


class RandomSearch:
    """Draw independent configurations from the defined parameter space."""

    def __init__(
        self,
        search_space: dict[str, Categorical | Distribution],
        *,
        seed: int = 0,
        include_baseline: bool = True,
    ):
        self.search_space = search_space
        self.seed = seed
        self.include_baseline = include_baseline
        self.index_offset = 0
        self.prior_best: float | None = None
        self.mlflow_tags: dict[str, str] = {}

    def ask(self, config_index: int) -> dict:
        if self.include_baseline and config_index == 0:
            return {}
        # per-index seeding keeps draws identical across restarts and concurrency
        rng = random.Random(f"{self.seed}-{config_index}")
        return {key: sample_one(spec, rng) for key, spec in self.search_space.items()}

    def tell(self, config_index: int, value: float | None) -> None:
        pass


def _to_optuna_distribution(spec: Categorical | Distribution):
    from optuna import distributions

    if isinstance(spec, Categorical):
        return distributions.CategoricalDistribution(spec.values)
    if spec.kind == "int":
        return distributions.IntDistribution(low=int(spec.low), high=int(spec.high))
    return distributions.FloatDistribution(
        low=spec.low, high=spec.high, log=spec.kind == "logfloat"
    )


def _baseline_params(
    search_space: dict[str, Categorical | Distribution], base_config: dict
) -> dict:
    """Extract valid base-config values for the searched keys."""
    params = {}
    for key, spec in search_space.items():
        value = read_dotted(base_config, key)
        if value is None:
            log.warning("baseline: %s not set in the base config, leaving it to the sampler", key)
            continue
        if isinstance(spec, Categorical):
            if value not in spec.values:
                log.warning("baseline: %s=%r is not one of %s, skipping", key, value, spec.values)
                continue
        else:
            if not spec.low <= value <= spec.high:
                log.warning(
                    "baseline: %s=%r is outside [%s, %s], skipping", key, value, spec.low, spec.high
                )
                continue
            value = int(value) if spec.kind == "int" else float(value)
        params[key] = value
    return params


def _study_key(
    search_space: dict[str, Categorical | Distribution],
    base_config: dict,
    direction: str,
    selection_metric: str,
) -> str:
    """Hash everything that makes trial scores comparable across runs."""
    # environment-specific keys are excluded so the same search resumes anywhere
    run = {
        k: v for k, v in base_config.get("run", {}).items() if k not in ("model_version", "model_dir")
    }
    data = {k: v for k, v in base_config.get("data", {}).items() if k != "data_dir"}
    comparable = {k: v for k, v in base_config.items() if k != "mlflow"} | {"run": run, "data": data}
    return optuna_utils.payload_hash(
        {
            "search_space": {key: spec.model_dump() for key, spec in search_space.items()},
            "base_config": comparable,
            "direction": direction,
            "selection_metric": selection_metric,
        }
    )


class OptunaSearch:
    """Propose configurations with an Optuna study updated as trials complete.

    Studies live in the storage resolved by ``optuna_utils.storage_url`` and are named
    ``<study_prefix>-<hash>-<NNN>``; the sweep passes the MLflow experiment name as prefix,
    so a study family belongs to one experiment. Identical searches share
    the hash; ``NNN`` counts the studies created for it. Each search starts a new study unless
    ``resume_study`` names an existing one, whose hash must match the current search.

    ``index_offset`` is the resumed study's trial count; the driver continues ``config_index``
    from there so trial names and Optuna trial numbers stay aligned. ``prior_best`` is the best
    value among the resumed study's completed trials, or ``None``.
    """

    def __init__(
        self,
        search_space: dict[str, Categorical | Distribution],
        base_config: dict,
        *,
        study_prefix: str,
        seed: int = 0,
        direction: str = "minimize",
        selection_metric: str = "val_loss",
        include_baseline: bool = True,
        resume_study: str | None = None,
        sampler: SamplerConfig | None = None,
        pruner: PrunerConfig | None = None,
    ):
        import optuna

        self._optuna = optuna
        self._distributions = {
            key: _to_optuna_distribution(spec) for key, spec in search_space.items()
        }
        storage = optuna_utils.storage_url()
        key = _study_key(search_space, base_config, direction, selection_metric)
        family = f"{study_prefix}-{key}"
        existing = optuna_utils.list_studies(storage, family)
        # pruner is not stored in study, so always recreate; extra config attributes are its kwargs
        pruner_instance = (
            optuna_utils.build_pruner(pruner.name, **(pruner.model_extra or {})) if pruner else None
        )
        resume = resume_study is not None
        # sampler is not stored in study either; on resume offset its seed by the prior trials
        # so it doesn't replay the same draws
        sampler_name = sampler.name if sampler else None
        sampler_kwargs = (sampler.model_extra or {}) if sampler else {}
        if resume_study is not None:
            resume_key, _ = optuna_utils.parse_study_name(resume_study, study_prefix)
            if resume_key != key:
                raise ValueError(
                    f"study {resume_study} belongs to a different search (hash {resume_key}, "
                    f"current {key}); the hash covers the search space, the base config except "
                    "[mlflow] and environment paths, direction and selection_metric"
                )
            if resume_study not in existing:
                raise ValueError(
                    f"no study {resume_study}; existing for this search: {sorted(existing)}"
                )
            n_prior = existing[resume_study].n_trials
            self._study = optuna_utils.load_study(
                storage,
                resume_study,
                sampler=optuna_utils.build_sampler(
                    sampler_name, seed=seed + n_prior, **sampler_kwargs
                ),
                pruner=pruner_instance,
            )
        else:
            n_prior = 0
            self._study = optuna_utils.create_numbered_study(
                storage,
                family,
                direction=direction,
                sampler=optuna_utils.build_sampler(sampler_name, seed=seed, **sampler_kwargs),
                pruner=pruner_instance,
            )
        study_name = self._study.study_name
        self.index_offset = n_prior
        self.mlflow_tags = {
            "optuna_study_name": study_name,
            "optuna_prior_trials": str(n_prior),
        }
        log.info(
            "optuna: %s study %s with %d prior trials",
            "resuming" if resume else "created", study_name, n_prior,
        )
        self.prior_best = optuna_utils.best_completed_value(self._study)
        if include_baseline and not resume:
            params = _baseline_params(search_space, base_config)
            if params:
                self._study.enqueue_trial(params)

    def ask(self, config_index: int) -> dict:
        trial = self._study.ask(self._distributions)
        if trial.number != config_index:
            raise RuntimeError(
                f"optuna trial number {trial.number} != config_index {config_index}; "
                "the study has stale waiting trials; start a new study instead of resuming it"
            )
        log.info("optuna: config_index=%d proposed %s", config_index, trial.params)
        return dict(trial.params)

    def tell(self, config_index: int, value: float | None) -> None:
        if value is None:
            self._study.tell(
                config_index, state=self._optuna.trial.TrialState.FAIL, skip_if_finished=True
            )
            log.info("optuna: config_index=%d failed", config_index)
        else:
            self._study.tell(config_index, value, skip_if_finished=True)
            log.info(
                "optuna: config_index=%d scored %.4f, best so far %.4f",
                config_index, value, self._study.best_value,
            )

    def report(self, config_index: int, history: dict[int, float]) -> int | None:
        """Report unseen epochs of a running trial; return the step to prune it at, if any.

        Does not close the trial: cancel its job first, then call ``tell_pruned``. Until then
        the trial stays running, so a failed cancel is retried by the next call.
        """
        running = optuna_utils.running_trial(self._study, config_index)
        if running is None:
            return None
        trial, frozen = running
        # a prune decided on an earlier call whose cancel failed; should_prune reads the stored steps
        if frozen.intermediate_values and trial.should_prune():
            return max(frozen.intermediate_values)
        for step in sorted(history):
            if step in frozen.intermediate_values:
                continue
            trial.report(history[step], step)
            if trial.should_prune():
                return step
        return None

    def tell_pruned(self, config_index: int) -> None:
        """Close a trial ``report`` chose to prune, once its job is cancelled."""
        self._study.tell(
            config_index, state=self._optuna.trial.TrialState.PRUNED, skip_if_finished=True
        )
        log.info("optuna: config_index=%d pruned", config_index)


def build_search_strategy(
    sweep: SweepConfig,
    strategy: StrategyConfig,
    search_space: dict[str, Categorical | Distribution],
    base_config: dict,
    *,
    resume_study: str | None = None,
) -> SearchStrategy:
    """Build the configured random or Optuna proposal strategy."""
    if resume_study is not None and strategy.name != "optuna":
        raise ValueError(f"resume_study only applies to the optuna strategy, not {strategy.name}")
    if strategy.name == "optuna":
        return OptunaSearch(
            search_space,
            base_config,
            study_prefix=base_config["mlflow"]["experiment_name"],
            seed=strategy.seed,
            direction=sweep.direction,
            selection_metric=sweep.selection_metric,
            include_baseline=sweep.include_baseline,
            resume_study=resume_study,
            sampler=strategy.sampler,
            pruner=strategy.pruner,
        )
    return RandomSearch(
        search_space, seed=strategy.seed, include_baseline=sweep.include_baseline
    )

__all__ = [
    "OptunaSearch",
    "RandomSearch",
    "SearchStrategy",
    "build_search_strategy",
]