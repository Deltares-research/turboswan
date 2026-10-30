"""Algorithms that propose the next configured parameter combination.

Random and Optuna proposals decide which values to try only within the
researcher-defined search space. They do not establish scientific validity or
acceptance criteria.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
import random
import re
from typing import Protocol, runtime_checkable

from turboswan.mlops.sweeps.configuration import (
    Categorical,
    Distribution,
    PrunerConfig,
    StrategyConfig,
    SweepConfig,
    sample_one,
)
from turboswan.training import read_dotted

log = logging.getLogger(__name__)

# TODO: replace by postgresql database
OPTUNA_STORAGE_DIR = pathlib.Path.home() / "turboswan" / "optuna"


@runtime_checkable
class SearchStrategy(Protocol):
    """One configuration out per ``ask`` and one score back per ``tell``."""

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
    payload = json.dumps(
        {
            "search_space": {key: spec.model_dump() for key, spec in search_space.items()},
            "base_config": comparable,
            "direction": direction,
            "selection_metric": selection_metric,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


class OptunaSearch:
    """Propose configurations with an Optuna study updated as trials complete.

    Studies live in ``TURBOSWAN_OPTUNA_STORAGE`` (default: sqlite under ``OPTUNA_STORAGE_DIR``)
    and are named ``turboswan-<hash>-<NNN>``. Identical searches share
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
        seed: int = 0,
        direction: str = "minimize",
        selection_metric: str = "val_loss",
        include_baseline: bool = True,
        resume_study: str | None = None,
        pruner: PrunerConfig | None = None,
    ):
        import optuna

        self._optuna = optuna
        self._distributions = {
            key: _to_optuna_distribution(spec) for key, spec in search_space.items()
        }
        storage = os.environ.get("TURBOSWAN_OPTUNA_STORAGE")
        if not storage:
            OPTUNA_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
            storage = f"sqlite:///{(OPTUNA_STORAGE_DIR / 'studies.db').as_posix()}"
        key = _study_key(search_space, base_config, direction, selection_metric)
        prefix = f"turboswan-{key}-"
        existing = {
            summary.study_name: summary
            for summary in optuna.get_all_study_summaries(storage)
            if summary.study_name.startswith(prefix)
        }
        # pruner is not stored in study, so always recreate
        pruner_instance = (
            # pass extra pruner config attributes to pruner class
            getattr(optuna.pruners, pruner.name)(**(pruner.model_extra or {}))
            if pruner
            else None
        )
        resume = resume_study is not None
        if resume:
            match = re.fullmatch(r"turboswan-([0-9a-f]{12})-\d+", resume_study)
            if match is None:
                raise ValueError(f"{resume_study!r} is not a study name like turboswan-<hash>-NNN")
            if match.group(1) != key:
                raise ValueError(
                    f"study {resume_study} belongs to a different search (hash {match.group(1)}, "
                    f"current {key}); the hash covers the search space, the base config except "
                    "[mlflow] and environment paths, direction and selection_metric"
                )
            if resume_study not in existing:
                raise ValueError(
                    f"no study {resume_study}; existing for this search: {sorted(existing)}"
                )
            study_name = resume_study
            n_prior = existing[study_name].n_trials
            # sampler is also not persisted, so offset seed by prior trials so it doesn't replay the same draws
            self._study = optuna.load_study(
                study_name=study_name,
                storage=storage,
                sampler=optuna.samplers.TPESampler(seed=seed + n_prior),
                pruner=pruner_instance,
            )
        else:
            n_prior = 0
            number = max((int(name.removeprefix(prefix)) for name in existing), default=0) + 1
            while number < 1000:  # arbitrary upper limit to avoid infinite loop
                study_name = f"{prefix}{number:03d}"
                try:
                    self._study = optuna.create_study(
                        storage=storage,
                        study_name=study_name,
                        direction=direction,
                        sampler=optuna.samplers.TPESampler(seed=seed),
                        pruner=pruner_instance,
                    )
                    break
                # a concurrent sweep of the same search took this number
                except optuna.exceptions.DuplicatedStudyError:
                    number += 1
            else:
                # arbitrary limit, but avoids infinite loop and > 1000 identical studies would be odd
                raise RuntimeError("could not create a unique study name after 1000 attempts")
        self.index_offset = n_prior
        self.mlflow_tags = {
            "optuna_study_name": study_name,
            "optuna_prior_trials": str(n_prior),
        }
        log.info(
            "optuna: %s study %s with %d prior trials",
            "resuming" if resume else "created", study_name, n_prior,
        )
        completed = [
            t.value
            for t in self._study.get_trials(
                deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,)
            )
        ]
        self.prior_best = (min if direction == "minimize" else max)(completed) if completed else None
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
        states = self._optuna.trial.TrialState
        # unfiltered get_trials is ordered by trial number, so the index is the number
        frozen = self._study.get_trials(deepcopy=False)[config_index]
        if frozen.state != states.RUNNING:
            return None
        # optuna has no public way to rebuild a Trial from a stored one; _trial_id is private
        trial = self._optuna.trial.Trial(self._study, frozen._trial_id)
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
            seed=strategy.seed,
            direction=sweep.direction,
            selection_metric=sweep.selection_metric,
            include_baseline=sweep.include_baseline,
            resume_study=resume_study,
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