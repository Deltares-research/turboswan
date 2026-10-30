"""MLflow helpers for demo notebooks and DAGs."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import mlflow
from mlflow.models import ModelSignature
from mlflow.tracking import MlflowClient


def configure_tracking(
    experiment_name: str,
    tracking_uri: str | None = None,
) -> str:
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(uri)
    try:
        mlflow.set_experiment(experiment_name)
    except mlflow.exceptions.MlflowException as exc:
        if "401" not in str(exc):
            raise
        user = os.environ.get("MLFLOW_TRACKING_USERNAME") or "(unset)"
        raise mlflow.exceptions.MlflowException(
            f"MLflow 401 as user {user!r} (URI {uri}). "
            "JupyterHub injects MLFLOW_TRACKING_USERNAME/PASSWORD from Secret "
            "mlflow-auth. If those are set in the pod, this tracking server's "
            "auth DB does not have that user — re-run Job mlflow-auth-bootstrap."
        ) from exc
    return uri


@contextmanager
def log_run(
    *,
    run_name: str,
    experiment_name: str,
    params: dict[str, Any],
    metrics: dict[str, float],
    airflow_run_id: str = "",
    tags: dict[str, str] | None = None,
    tracking_uri: str | None = None,
) -> Iterator[str]:
    """Open an MLflow run, log params/metrics, yield run_id."""
    configure_tracking(experiment_name, tracking_uri)
    with mlflow.start_run(run_name=run_name):
        if airflow_run_id:
            mlflow.set_tag("airflow_run_id", airflow_run_id)
        if tags:
            mlflow.set_tags(tags)
        mlflow.log_params({k: str(v) for k, v in params.items()})
        mlflow.log_metrics(metrics)
        run_id = mlflow.active_run().info.run_id
        yield run_id


def _default_code_paths(model) -> list[str]:
    """Parent dir of the model's top-level package (e.g. …/dune_ai/src).

    Never point this at the SDK — serving unpickles the *project* module.
    """
    import importlib

    mod_name = type(model).__module__  # e.g. dune_ai.baseline_rf
    root = mod_name.split(".", 1)[0]
    if root in ("builtins", "__main__", "sklearn"):
        raise ValueError(
            "register_sklearn_model needs code_paths= for this model "
            f"(module={mod_name!r}); pass the project package src dir"
        )
    pkg = importlib.import_module(root)
    if not getattr(pkg, "__file__", None):
        raise ValueError(f"cannot resolve code_paths for package {root!r}")
    # …/src/<pkg>/__init__.py → …/src
    return [str(Path(pkg.__file__).resolve().parent.parent)]


# Estimators that need their own package installed to serve, unlike sklearn's own.
_NON_SKLEARN_ESTIMATOR_LIBS = ("xgboost", "lightgbm", "catboost")


def _default_extra_pip_requirements() -> list[str]:
    """Pin whichever of xgboost/lightgbm/catboost is installed here.

    Without this, the serving container may not install them and fail to load
    the model.
    """
    import importlib.metadata

    reqs = []
    for lib in _NON_SKLEARN_ESTIMATOR_LIBS:
        try:
            reqs.append(f"{lib}=={importlib.metadata.version(lib)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    return reqs


def register_sklearn_model(
    model,
    *,
    registered_model_name: str | None,
    input_example,
    signature: ModelSignature | None = None,
    code_paths: list[str] | None = None,
    extra_pip_requirements: list[str] | None = None,
) -> str:
    """Log a sklearn-compatible model (pickle + code_paths for KServe).

    ``registered_model_name=None`` logs without creating a registry version
    (sweeps: register only the winner later). Deliberately keeps the exact same
    ``serialization_format``/``code_paths`` in both cases; logging without
    ``code_paths`` silently broke KServe unpickling.

    ``code_paths`` must be the *project* package src dir (e.g. dune_ai's
    ``src/``), never this SDK. Defaults to the parent of the model's defining
    package. Keep paths free of a nested ``.git`` so MLflow 3's code snapshot
    stays flat for the older MLServer client (2.10.2).

    ``extra_pip_requirements`` defaults to pinning xgboost/lightgbm/catboost
    if installed (see ``_default_extra_pip_requirements``). Pass ``[]`` to
    skip that.
    """
    import mlflow.sklearn  # local: sklearn/scipy must not load at DAG parse time

    paths = code_paths if code_paths is not None else _default_code_paths(model)
    extra_reqs = (
        extra_pip_requirements
        if extra_pip_requirements is not None
        else _default_extra_pip_requirements()
    )
    info = mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=input_example,
        registered_model_name=registered_model_name,
        serialization_format="pickle",
        code_paths=paths,
        extra_pip_requirements=extra_reqs or None,
    )
    return info.model_uri


def register_model_from_run(client: MlflowClient, model_name: str, run_id: str) -> str:
    """Register the ``model`` artifact of an already-logged (unregistered) run.

    Used to promote only the winning trial of a sweep to the registry, after
    every trial logged its model unregistered via ``register_sklearn_model``.
    """
    mv = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    return str(mv.version)


def model_version_for_run(client: MlflowClient, model_name: str, run_id: str) -> str:
    """Return the registry version string for a given MLflow run."""
    versions = [v for v in client.search_model_versions(f"name='{model_name}'") if v.run_id == run_id]
    if not versions:
        raise RuntimeError(
            f"No registered model version for run {run_id!r} under {model_name!r}"
        )
    return str(versions[0].version)


def search_runs_for_airflow_dag(
    client: MlflowClient,
    experiment_name: str,
    airflow_run_id: str,
    *,
    run_type: str = "model",
    metric: str = "rmse",
    order_by: list[str] | None = None,
) -> list:
    """Return MLflow runs tagged with the given Airflow dag_run id."""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name!r}")
    if order_by is None:
        order_by = [f"metrics.{metric} ASC"]
    return client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            f"tags.`airflow_run_id` = '{airflow_run_id}' AND tags.run_type = '{run_type}'"
        ),
        order_by=order_by,
        max_results=100,
    )


def tracking_client(tracking_uri: str | None = None) -> MlflowClient:
    """Configured ``MlflowClient`` for DAG tasks and drivers."""
    uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(uri)
    return MlflowClient()


def ensure_experiment_id(client: MlflowClient, experiment_name: str) -> str:
    """Return an experiment id, creating the experiment if needed."""
    exp = client.get_experiment_by_name(experiment_name)
    if exp is not None:
        return exp.experiment_id
    try:
        return client.create_experiment(experiment_name)
    except Exception:
        exp = client.get_experiment_by_name(experiment_name)
        if exp is None:
            raise
        return exp.experiment_id


def get_or_create_sweep_parent_run(
    client: MlflowClient,
    experiment_name: str,
    airflow_run_id: str,
    *,
    n_configs: int,
    project: str,
    extra_params: dict[str, str] | None = None,
    run_name: str | None = None,
) -> str:
    """Pre-create a sweep parent run, reusing it on Airflow task retry."""
    exp_id = ensure_experiment_id(client, experiment_name)
    existing = client.search_runs(
        experiment_ids=[exp_id],
        filter_string=(
            f"tags.`airflow_run_id` = '{airflow_run_id}' AND tags.run_type = 'sweep'"
        ),
        order_by=["attributes.start_time ASC"],
        max_results=1,
    )
    if existing:
        return existing[0].info.run_id

    run = client.create_run(
        experiment_id=exp_id,
        run_name=run_name or f"sweep-{airflow_run_id[:8]}-n{n_configs}",
        tags={
            "airflow_run_id": airflow_run_id,
            "project": project,
            "run_type": "sweep",
            "n_configs": str(n_configs),
        },
    )
    if extra_params:
        client.log_params(run.info.run_id, extra_params)
    return run.info.run_id


def latest_runs_by_config(
    client: MlflowClient,
    experiment_name: str,
    airflow_run_id: str,
    *,
    run_type: str = "model",
    expected_configs: int | None = None,
) -> dict[str, Any]:
    """Return the latest run per ``config_index`` for an Airflow DAG execution."""
    runs = search_runs_for_airflow_dag(
        client,
        experiment_name,
        airflow_run_id,
        run_type=run_type,
        order_by=["attributes.start_time DESC"],
    )
    by_config: dict[str, Any] = {}
    for run in runs:
        idx = run.data.tags.get("config_index")
        if idx is not None and idx not in by_config:
            by_config[idx] = run

    if expected_configs is not None:
        missing = {str(i) for i in range(expected_configs)} - set(by_config)
        if missing:
            raise RuntimeError(
                f"No MLflow run for config_index: {sorted(missing)} "
                f"(airflow_run_id={airflow_run_id!r})"
            )
    return by_config


def sweep_runs(
    client: MlflowClient,
    experiment_name: str,
    airflow_run_id: str,
    *,
    expected_configs: int,
    metric: str = "rmse",
) -> list:
    """Like ``search_runs_for_airflow_dag``, but deduped to one run per
    ``config_index`` tag (keeps the lowest-``metric`` run per config).

    Ray retries a trial killed by a Spot eviction, which logs a *second* MLflow
    run for the same ``config_index`` — so a raw run count no longer equals
    ``expected_configs``. Relies on ``search_runs_for_airflow_dag`` already
    ordering by ``metric`` ascending, so the first run seen per config via
    ``setdefault`` is the best one.
    """
    runs = search_runs_for_airflow_dag(client, experiment_name, airflow_run_id, metric=metric)
    by_config: dict[str, Any] = {}
    for run in runs:
        by_config.setdefault(run.data.tags.get("config_index"), run)

    missing = {str(i) for i in range(expected_configs)} - by_config.keys()
    if missing:
        raise RuntimeError(
            f"No MLflow run for config_index: {sorted(missing)} "
            f"(airflow_run_id={airflow_run_id!r})"
        )
    return list(by_config.values())


def best_run_for_airflow_dag(
    client: MlflowClient,
    experiment_name: str,
    airflow_run_id: str,
    *,
    expected_runs: int | None = None,
    metric: str = "rmse",
) -> Any:
    """Pick the lowest-RMSE model run for an Airflow DAG execution."""
    runs = search_runs_for_airflow_dag(client, experiment_name, airflow_run_id)
    if expected_runs is not None and len(runs) != expected_runs:
        raise RuntimeError(
            f"Expected {expected_runs} MLflow runs for airflow_run_id={airflow_run_id!r}, "
            f"found {len(runs)}"
        )
    if not runs:
        raise RuntimeError(f"No MLflow model runs for airflow_run_id={airflow_run_id!r}")

    missing_metric = [r.info.run_id for r in runs if metric not in r.data.metrics]
    if missing_metric:
        raise RuntimeError(
            f"MLflow runs missing metric {metric!r}: {', '.join(missing_metric)}"
        )
    return min(runs, key=lambda run: run.data.metrics[metric])


def promote_model_version_to_production(
    client: MlflowClient,
    model_name: str,
    *,
    version: str | None = None,
    run_id: str | None = None,
    archive_existing_versions: bool = True,
) -> str:
    """Move a registry version to Production (archives the previous Production version)."""
    if version is None:
        if run_id is None:
            raise ValueError("provide version or run_id")
        version = model_version_for_run(client, model_name, run_id)
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production",
        archive_existing_versions=archive_existing_versions,
    )
    return version
