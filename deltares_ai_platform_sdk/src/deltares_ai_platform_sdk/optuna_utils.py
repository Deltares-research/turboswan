"""Optuna helpers for sweeps: storage, numbered studies, pruners and trials."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import optuna


def storage_url(storage: str | None = None) -> str:
    """Resolve the Optuna storage URL.

    Order: ``storage`` argument, then the ``OPTUNA_STORAGE`` environment variable,
    then a sqlite ``studies.db`` in ``~/.optuna``, which is created if needed.
    """
    url = storage or os.environ.get("OPTUNA_STORAGE")
    if url:
        return url
    # no storage url set, default to local sqlite db
    directory = Path.home() / ".optuna"
    directory.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{(directory / 'studies.db').as_posix()}"


def payload_hash(payload: Any, length: int = 12) -> str:
    """Short, stable sha256 of a JSON-serializable payload, e.g. to name a study family."""
    text = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:length]


def build_pruner(name: str | None, **kwargs: Any) -> optuna.pruners.BasePruner | None:
    """Instantiate ``optuna.pruners.<name>(**kwargs)``, or ``None`` without a name."""
    if name is None:
        return None
    return getattr(optuna.pruners, name)(**kwargs)


def build_sampler(name: str | None = None, **kwargs: Any) -> optuna.samplers.BaseSampler:
    """Instantiate ``optuna.samplers.<name>(**kwargs)``; ``TPESampler`` without a name.

    Samplers are not persisted with a study, so build a new one whenever a study is
    created or loaded. When resuming, offset ``seed`` by the number of prior trials so
    the sampler does not replay the same draws.
    """
    return getattr(optuna.samplers, name or "TPESampler")(**kwargs)


def list_studies(storage: str, family: str) -> dict[str, optuna.study.StudySummary]:
    """Return the summaries of studies named ``<family>-NNN``, keyed by study name."""
    return {
        summary.study_name: summary
        for summary in optuna.get_all_study_summaries(storage)
        if summary.study_name.startswith(f"{family}-")
    }


def parse_study_name(name: str, prefix: str) -> tuple[str, int]:
    """Split ``<prefix>-<hex hash>-NNN`` into ``(hash, NNN)``; raise ``ValueError`` otherwise."""
    match = re.fullmatch(rf"{re.escape(prefix)}-([0-9a-f]+)-(\d+)", name)
    if match is None:
        raise ValueError(f"{name!r} is not a study name like {prefix}-<hash>-NNN")
    return match.group(1), int(match.group(2))


def create_numbered_study(
    storage: str,
    family: str,
    *,
    direction: str = "minimize",
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
    max_number: int = 999,
) -> optuna.Study:
    """Create the next free study ``<family>-NNN``, numbering from the highest existing one.

    Retries with the next number if a concurrent process took it first.
    """
    numbers = [int(name.rsplit("-", 1)[1]) for name in list_studies(storage, family)]
    number = max(numbers, default=0) + 1
    while number <= max_number:
        try:
            return optuna.create_study(
                storage=storage,
                study_name=f"{family}-{number:03d}",
                direction=direction,
                sampler=sampler,
                pruner=pruner,
            )
        except optuna.exceptions.DuplicatedStudyError:
            number += 1
    # avoids an infinite loop; that many identical studies would be odd anyway
    raise RuntimeError(f"could not create a unique study name for {family} below {max_number + 1}")


def load_study(
    storage: str,
    study_name: str,
    *,
    sampler: optuna.samplers.BaseSampler | None = None,
    pruner: optuna.pruners.BasePruner | None = None,
) -> optuna.Study:
    """Load an existing study with the given sampler and pruner (neither is persisted)."""
    return optuna.load_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
    )


def best_completed_value(study: optuna.Study) -> float | None:
    """Best value among the study's completed trials, or ``None`` if there are none."""
    values = [
        t.value
        for t in study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if t.value is not None
    ]
    if not values:
        return None
    return min(values) if study.direction == optuna.study.StudyDirection.MINIMIZE else max(values)


def running_trial(
    study: optuna.Study, number: int
) -> tuple[optuna.trial.Trial, optuna.trial.FrozenTrial] | None:
    """Rebuild the live ``Trial`` for a running trial number, with its stored state.

    Returns ``None`` if the trial is no longer running.
    """
    # unfiltered get_trials is ordered by trial number, so the index is the number
    frozen = study.get_trials(deepcopy=False)[number]
    if frozen.state != optuna.trial.TrialState.RUNNING:
        return None
    # optuna has no public way to rebuild a Trial from a stored one; _trial_id is private
    return optuna.trial.Trial(study, frozen._trial_id), frozen


__all__ = [
    "best_completed_value",
    "build_pruner",
    "build_sampler",
    "create_numbered_study",
    "list_studies",
    "load_study",
    "parse_study_name",
    "payload_hash",
    "running_trial",
    "storage_url",
]
