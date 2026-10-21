"""Work out what to write down about a run: its name, settings, tags and notes.

Everything here is plain bookkeeping. It reads the recipe and decides which
values become MLflow parameters, how they are named, and checks the names are
ones MLflow's naming convention accepts. It never touches the training itself.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from turboswan import training

# Letters, digits and ``_ . / -`` plus spaces. MLflow also accepts ``:`` outside
# Windows, but rejects it there, so it is excluded to keep recipes portable.
# Anything else (for example ``%`` or ``#``) is rejected up front rather than
# failing inside MLflow.
MLFLOW_NAME_PATTERN = r"^[A-Za-z0-9_. /\-]+$"

ParamValue = str | int | float | bool | None


@dataclass(frozen=True)
class RunMetadata:
    """Everything about a run, other than the model itself, ready for MLflow."""

    experiment_name: str
    run_name: str
    params: dict[str, ParamValue]
    tags: dict[str, str]
    description: str


def validate_identifiers(**identifier_groups: Iterable[str]) -> None:
    """Stop early if any name uses characters MLflow's convention disallows.

    Each keyword is a group label and its value the names to check, e.g.
    ``validate_identifiers(metric_names=["loss"], tag_names=["project"])``.
    The error lists every offending name, grouped by label, so all problems
    can be fixed in one go.
    """
    pattern = re.compile(MLFLOW_NAME_PATTERN)
    invalid_identifiers = {
        group: [name for name in names if not pattern.fullmatch(name)]
        for group, names in identifier_groups.items()
    }
    # Keep only the groups that actually have a problem.
    invalid_identifiers = {
        group: names for group, names in invalid_identifiers.items() if names
    }
    if invalid_identifiers:
        details = "; ".join(
            f"{group}: {', '.join(names)}"
            for group, names in invalid_identifiers.items()
        )
        raise ValueError(
            f"MLflow identifiers contain unsupported characters: {details}"
        )


def label_names(names: Iterable[str], labels: Mapping[str, str]) -> dict[str, str]:
    """Map each internal name to the readable label the recipe gives it.

    Names without a label keep their own name, so nothing is silently dropped.
    Two different names may not end up with the same label, since one would
    overwrite the other in MLflow.

    Returns:
        Dictionary from original name to the name to record.
    """
    resolved = {name: labels.get(name, name) for name in names}
    validate_identifiers(names=resolved.values())
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("Tracking labels must not map different names to the same label")
    return resolved


def portable_path(path: str | Path, *, relative_to: Path | None = None) -> str:
    """Shorten a path so recorded runs do not expose personal folder names.

    A path inside ``relative_to`` becomes relative to it; otherwise a path
    inside the home folder is written with ``~``. Any other path is unchanged.
    """
    path = Path(path)
    if relative_to is not None and path.is_relative_to(relative_to):
        return str(path.relative_to(relative_to))
    if path.is_absolute() and path.is_relative_to(Path.home()):
        return str(Path("~") / path.relative_to(Path.home()))
    return str(path)


def recorded_config(config: training.TrainConfig) -> dict[str, object]:
    """Return the recipe as a plain dictionary with portable paths.

    This is the version saved to MLflow. The recipe in memory, which training
    keeps using, is left untouched.
    """
    def serialize(value: object) -> object:
        # Walk through nested dictionaries and lists, converting every Path.
        if isinstance(value, Path):
            return portable_path(value)
        if isinstance(value, dict):
            return {key: serialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [serialize(item) for item in value]
        return value

    snapshot = {key: serialize(value) for key, value in config.model_dump(mode="python").items()}
    snapshot["data"]["source"] = portable_path(config.data.source)
    return snapshot


def recorded_data_source(config: training.TrainConfig, location: Path) -> str:
    """Say where a data folder came from, as a shortened local path."""
    return portable_path(location)


def flatten_config(config: training.TrainConfig) -> dict[str, ParamValue]:
    """Turn the recipe into a flat list of settings that MLflow can store.

    MLflow parameters are single name/value pairs, but the recipe is nested.
    Nested sections are joined with dots (``training.epochs``), lists are
    written as JSON text, and anything unusual becomes text. The ``[mlflow]``
    section itself is skipped: it describes how to record, not what was trained.
    """
    params: dict[str, ParamValue] = {}

    def add_values(values: Mapping[str, object], prefix: str = "") -> None:
        for key, value in values.items():
            parameter_name = f"{prefix}.{key}" if prefix else key
            if isinstance(value, Mapping):
                add_values(value, parameter_name)
            elif isinstance(value, list):
                params[parameter_name] = json.dumps(value)
            elif value is None or isinstance(value, (str, int, float, bool)):
                params[parameter_name] = value
            else:
                params[parameter_name] = str(value)

    add_values({key: value for key, value in recorded_config(config).items() if key != "mlflow"})
    return params


def describe_variables(
    variables: Iterable[str], descriptions: Mapping[str, str]
) -> str:
    """Write variable names with their meaning, e.g. ``hs (wave height)``.

    A variable without a description is listed by name only.
    """
    return ", ".join(
        f"{variable} ({descriptions[variable]})" if variable in descriptions else variable
        for variable in variables
    )


def tfrecord_manifest_digest(files: Iterable[str]) -> str:
    """Make a short fingerprint of a list of data files.

    The same file names always give the same fingerprint, whatever their order
    or folder, so runs on the same shards can be recognised as such. It is
    shortened to 36 characters, the longest digest MLflow accepts.
    """
    manifest = "\n".join(sorted(Path(file).name for file in files))
    return hashlib.sha256(manifest.encode()).hexdigest()[:36]


def build_run_metadata(
    config: training.TrainConfig,
    *,
    experiment_name: str,
    run_name: str | None = None,
    description: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> RunMetadata:
    """Assemble the name, settings, tags and description recorded for a run.

    Settings come from the recipe, so they are never typed out twice. Optional
    ``[mlflow]`` sections in the recipe then shape what is shown:

    * ``variable_descriptions`` adds the meaning next to each variable name.
    * ``parameter_labels`` works as a *shortlist*: when present, only the
      settings it names are recorded, under the labels it gives.
    * ``tags`` are added to the standard ``project`` and ``run_type`` tags.

    Args:
        config: Validated recipe.
        experiment_name: The experiment (study) the run belongs to.
        run_name: Run name; defaults to the model version plus a UTC timestamp.
        description: Note shown on the run; a generic sentence if omitted.
        tags: Extra tags. They win over the standard ones on a clash.

    Raises:
        ValueError: If any resulting name has characters MLflow does not accept.
    """
    run_name = run_name or (
        f"turboswan-{config.run.model_version}-"
        f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    )
    params = flatten_config(config)
    # Replace the recipe's data.source with the form worth keeping: the S3
    # address, or a shortened local path (never a personal home folder).
    params.update(
        {
            "data.source": recorded_data_source(config, config.data.data_dir),
            "data.resolved_train_dir": portable_path(config.data.train_subdir),
            "data.resolved_val_dir": portable_path(config.data.val_subdir),
        }
    )
    if config.mlflow is not None:
        validate_identifiers(metric_names=config.mlflow.metric_labels.values())
        descriptions = config.mlflow.variable_descriptions
        if descriptions:
            for field in ("input_variables", "output_variables", "boundary_variables"):
                params[f"data.{field}"] = describe_variables(
                    getattr(config.data, field), descriptions
                )
        labels = config.mlflow.parameter_labels
        if labels:
            # Shortlist: drop everything the recipe did not label.
            params = {name: value for name, value in params.items() if name in labels}
        names = label_names(params, labels)
        params = {names[name]: value for name, value in params.items()}
    merged_tags = {"project": "turboswan", "run_type": "training"}
    if tags:
        merged_tags.update(tags)
    description = description or (
        "Tracked TurboSWAN training run. The model, configuration, preprocessing "
        "inputs, learning curve, and TFRecord manifests are stored as MLflow records."
    )
    validate_identifiers(
        experiment_names=[experiment_name],
        run_names=[run_name],
        parameter_names=params.keys(),
        tag_names=[*merged_tags, "mlflow.note.content"],
    )
    return RunMetadata(
        experiment_name=experiment_name,
        run_name=run_name,
        params=params,
        tags=merged_tags,
        description=description,
    )