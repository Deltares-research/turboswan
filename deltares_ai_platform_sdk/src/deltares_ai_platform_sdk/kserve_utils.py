"""KServe helpers — deploy InferenceService from MLflow models stored in MinIO/S3."""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from mlflow.tracking import MlflowClient

log = logging.getLogger(__name__)

KSERVE_GROUP = "serving.kserve.io"
KSERVE_VERSION = "v1beta1"
KSERVE_PLURAL = "inferenceservices"


def runtime_namespace() -> str:
    """Namespace the Airflow scheduler pod runs in (platform or pr-*)."""
    ns_file = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
    if os.path.isfile(ns_file):
        with open(ns_file, encoding="utf-8") as fh:
            return fh.read().strip()
    return os.environ.get("AIRFLOW__KUBERNETES__NAMESPACE", "platform")


def _exp_models_prefix(client: MlflowClient, run_id: str) -> str:
    """``s3://…/mlflow-artifacts/<exp_id>`` — parent of run and logged-model dirs."""
    run = client.get_run(run_id)
    run_root = run.info.artifact_uri.rstrip("/").removesuffix("/artifacts")
    return run_root.rsplit("/", 1)[0]


def _logged_model_id(client: MlflowClient, run_id: str) -> str | None:
    """MLflow 3 model id (``m-…``) for a run — tag first, then search."""
    run = client.get_run(run_id)
    tagged = (run.data.tags or {}).get("logged_model_id")
    if tagged:
        return tagged
    search = getattr(client, "search_logged_models", None)
    if search is None:
        return None
    try:
        page = search(
            experiment_ids=[run.info.experiment_id],
            filter_string=f"source_run_id = '{run_id}'",
            max_results=20,
        )
        matches = [m for m in page if getattr(m, "source_run_id", None) == run_id]
        if not matches:
            page = search(experiment_ids=[run.info.experiment_id], max_results=100)
            matches = [m for m in page if getattr(m, "source_run_id", None) == run_id]
    except Exception as exc:
        log.warning("search_logged_models failed for run %s: %s", run_id, exc)
        return None
    if not matches:
        return None
    best = max(matches, key=lambda m: getattr(m, "creation_timestamp", 0) or 0)
    return best.model_id


def _logged_model_storage_uri(client: MlflowClient, run_id: str, model_id: str) -> str:
    get_lm = getattr(client, "get_logged_model", None)
    if get_lm is not None:
        try:
            loc = get_lm(model_id).artifact_location
            if loc:
                return loc.rstrip("/")
        except Exception:
            pass
    return f"{_exp_models_prefix(client, run_id)}/models/{model_id}/artifacts"


def _source_to_storage_uri(client: MlflowClient, source: str, *, run_id: str | None = None) -> str:
    """Map registry ``source`` to an s3:// URI KServe can download.

    MLflow 3 ``log_model(name=…)`` stores under ``…/models/m-<id>/artifacts``,
    not under the run's ``artifacts/model``.
    """
    if source.startswith("s3://"):
        return source
    if source.startswith("models:/"):
        if not run_id:
            raise RuntimeError(f"Cannot resolve {source!r} without run_id")
        model_id = source[len("models:/") :].split("/", 1)[0]
        return _logged_model_storage_uri(client, run_id, model_id)
    if source.startswith("runs:/"):
        run_id_part, _, subpath = source[len("runs:/") :].partition("/")
        if not subpath or subpath == "model":
            model_id = _logged_model_id(client, run_id_part)
            if model_id:
                return _logged_model_storage_uri(client, run_id_part, model_id)
            # No MLflow 3 logged model — classic registry usage (create_model_version
            # with a runs:/ source and no log_model() call, e.g. the AIWAM DAGs).
            # "model" is then a plain run-relative artifact path, not a logged-model id.
            run = client.get_run(run_id_part)
            return f"{run.info.artifact_uri.rstrip('/')}/model"
        run = client.get_run(run_id_part)
        return f"{run.info.artifact_uri.rstrip('/')}/{subpath}"
    raise RuntimeError(f"Unsupported registered model source URI: {source!r}")


def _model_version_storage_uri(client: MlflowClient, version) -> str:
    return _source_to_storage_uri(client, version.source, run_id=version.run_id)



def registered_model_storage_uri(
    client: MlflowClient,
    model_name: str,
    *,
    run_id: str | None = None,
) -> str:
    """Resolve a registered MLflow model version to an s3:// storageUri for KServe."""
    versions = list(client.search_model_versions(f"name='{model_name}'"))
    if run_id:
        versions = [v for v in versions if v.run_id == run_id]
    if not versions:
        raise RuntimeError(f"No registered model versions found for {model_name!r}")
    version = max(versions, key=lambda v: int(v.version))
    return _model_version_storage_uri(client, version)


def production_model_storage_uri(client: MlflowClient, model_name: str) -> str:
    """Resolve the Production-stage model version to an s3:// storageUri for KServe."""
    production = [
        v for v in client.search_model_versions(f"name='{model_name}'")
        if v.current_stage == "Production"
    ]
    if not production:
        raise RuntimeError(
            f"No Production version for {model_name!r} — promote a version before deploying"
        )
    version = max(production, key=lambda v: int(v.version))
    uri = _model_version_storage_uri(client, version)
    log.info("Production %s v%s → %s", model_name, version.version, uri)
    return uri


def deploy_production_model_to_kserve(
    model_name: str,
    *,
    inference_service_name: str | None = None,
    namespace: str | None = None,
    client: MlflowClient | None = None,
    manifest_fn: Callable[..., dict[str, Any]] | None = None,
    **manifest_kwargs: Any,
) -> str:
    """Deploy (or update) InferenceService from the Production model in MLflow registry.

    ``manifest_fn`` defaults to :func:`inference_service_manifest` (mlflow/mlserver);
    pass e.g. :func:`custom_container_inference_service_manifest` for a non-mlflow-flavor
    predictor, with the extra kwargs it needs (e.g. ``image=...``) via ``manifest_kwargs``.
    """
    mlflow_client = client or MlflowClient()
    isvc_name = inference_service_name or model_name
    storage_uri = production_model_storage_uri(mlflow_client, model_name)
    build_manifest = manifest_fn or inference_service_manifest
    manifest = build_manifest(isvc_name, storage_uri, **manifest_kwargs)
    log.info("Deploying InferenceService %s storageUri=%s", isvc_name, storage_uri)
    upsert_inference_service(manifest, namespace=namespace)
    return storage_uri


# Demo endpoints idle almost all the time: measured 1-3m CPU steady state, so the
# cpu request only needs to cover scheduling, with the limit carrying inference
# burst. Memory is the opposite — it must cover the resident model, because a
# request below actual usage leaves the pod Burstable and eviction-eligible under
# node memory pressure (mlserver measured 371Mi; the Julia custom container 961Mi,
# hence the larger default on that path).
#
# KServe here runs in Standard mode with HPA and no KEDA, so minReplicas cannot go
# below 1 — scale-to-zero needs KEDA (+ its OTel scaler, which KServe's
# inferenceservice-config already points at) or Knative. Set explicitly anyway so
# the floor is a decision in the repo rather than a KServe default.
DEFAULT_PREDICTOR_MIN_REPLICAS = 1
DEFAULT_PREDICTOR_MAX_REPLICAS = 2
DEFAULT_PREDICTOR_RESOURCES: dict[str, dict[str, str]] = {
    "requests": {"cpu": "50m", "memory": "512Mi"},
    "limits": {"cpu": "1", "memory": "2Gi"},
}
DEFAULT_CUSTOM_PREDICTOR_RESOURCES: dict[str, dict[str, str]] = {
    "requests": {"cpu": "50m", "memory": "1Gi"},
    "limits": {"cpu": "1", "memory": "2Gi"},
}


def inference_service_manifest(
    name: str,
    storage_uri: str,
    *,
    secret_name: str = "storage-config-deltares",
    model_format: str = "mlflow",
    min_replicas: int = DEFAULT_PREDICTOR_MIN_REPLICAS,
    max_replicas: int = DEFAULT_PREDICTOR_MAX_REPLICAS,
    resources: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build an InferenceService spec that pulls model weights from S3."""
    return {
        "apiVersion": f"{KSERVE_GROUP}/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "annotations": {
                "serving.kserve.io/secretName": secret_name,
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "model": {
                    "modelFormat": {"name": model_format, "version": "1"},
                    "runtime": "kserve-mlserver",
                    "storageUri": storage_uri,
                    "resources": resources or DEFAULT_PREDICTOR_RESOURCES,
                },
            },
        },
    }


def custom_container_inference_service_manifest(
    name: str,
    storage_uri: str,
    *,
    image: str,
    args: list[str] | None = None,
    image_pull_policy: str = "Always",
    secret_name: str = "storage-config-deltares",
    port: int = 8080,
    min_replicas: int = DEFAULT_PREDICTOR_MIN_REPLICAS,
    max_replicas: int = DEFAULT_PREDICTOR_MAX_REPLICAS,
    resources: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build an InferenceService spec for a custom-container predictor.

    For models with no KServe-supported ``modelFormat`` (e.g. a Julia model with
    no mlflow flavor). Neither a bare ``predictor.storageUri`` field (not valid
    alongside ``containers`` on this KServe version's CRD — silently pruned) nor
    the ``serving.kserve.io/storageInitializerSourceUri`` annotation triggers
    storage-initializer injection here. What does: a container env var literally
    named ``STORAGE_URI`` — KServe's controller recognizes that name specifically,
    injects its real ``kserve/storage-initializer`` init container with this
    value as the source, downloads it to ``/mnt/models``, and rewrites the
    container's own ``STORAGE_URI`` to that local path afterward. So the app
    itself never needs its own S3 download code — just don't rely on
    ``STORAGE_URI``'s *value* for anything, only its presence.
    The injected storage-initializer gets its S3 credentials/endpoint from the
    ``serving.kserve.io/secretName`` annotation (same secret and mechanism
    :func:`inference_service_manifest`'s mlflow/mlserver path already uses) —
    without it the storage-initializer has no credentials and fails outright.
    """
    return {
        "apiVersion": f"{KSERVE_GROUP}/v1beta1",
        "kind": "InferenceService",
        "metadata": {
            "name": name,
            "annotations": {
                "serving.kserve.io/secretName": secret_name,
            },
        },
        "spec": {
            "predictor": {
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "containers": [
                    {
                        "name": "kserve-container",
                        "image": image,
                        "imagePullPolicy": image_pull_policy,
                        "args": args or ["serve"],
                        "ports": [{"containerPort": port, "protocol": "TCP"}],
                        "env": [
                            {"name": "MODEL_DIR", "value": "/mnt/models"},
                            {"name": "STORAGE_URI", "value": storage_uri},
                            # Must match `name` (the InferenceService/Service name
                            # predictor_url() builds the request path from), not
                            # whatever model_name happens to be in the trained
                            # model_settings.toml — serve.jl's V2 routes are
                            # registered under this, and the client always
                            # requests /v2/models/<isvc-name>/infer.
                            {"name": "MODEL_NAME", "value": name},
                        ],
                        # Julia predictor: measured 961Mi resident, so the memory
                        # request is 1Gi here rather than the mlserver path's 512Mi.
                        "resources": resources or DEFAULT_CUSTOM_PREDICTOR_RESOURCES,
                    },
                ],
            },
        },
    }


def _custom_objects_api():
    from kubernetes import client, config

    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()
    return client.CustomObjectsApi()


def upsert_inference_service(
    manifest: dict[str, Any],
    *,
    namespace: str | None = None,
) -> None:
    """Create or update an InferenceService (e.g. after a new MLflow model version)."""
    from kubernetes.client.exceptions import ApiException

    ns = namespace or runtime_namespace()
    name = manifest["metadata"]["name"]
    api = _custom_objects_api()
    try:
        existing = api.get_namespaced_custom_object(KSERVE_GROUP, KSERVE_VERSION, ns, KSERVE_PLURAL, name)
    except ApiException as exc:
        if exc.status != 404:
            raise
        log.info("Creating InferenceService %s in namespace %s", name, ns)
        api.create_namespaced_custom_object(KSERVE_GROUP, KSERVE_VERSION, ns, KSERVE_PLURAL, manifest)
        return
    manifest["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
    log.info("Updating InferenceService %s in namespace %s", name, ns)
    api.replace_namespaced_custom_object(KSERVE_GROUP, KSERVE_VERSION, ns, KSERVE_PLURAL, name, manifest)


def predictor_url(name: str, *, namespace: str | None = None) -> str:
    """MLServer V2 infer endpoint (cluster DNS)."""
    ns = namespace or runtime_namespace()
    return f"http://{name}-predictor.{ns}.svc.cluster.local/v2/models/{name}/infer"


def invoke_predictor(
    name: str,
    features: Mapping[str, list[float]],
    *,
    namespace: str | None = None,
    timeout_seconds: int = 60,
) -> list[float]:
    """Run tabular inference via KServe MLServer V2 (one list per feature column)."""
    if not features:
        raise ValueError("features must not be empty")
    n_rows = len(next(iter(features.values())))
    if n_rows == 0:
        raise ValueError("features must contain at least one row")
    for col, values in features.items():
        if len(values) != n_rows:
            raise ValueError(f"column {col!r} has {len(values)} values, expected {n_rows}")

    payload = {
        "inputs": [
            {
                "name": col,
                "shape": [n_rows],
                "datatype": "FP64",
                "data": [float(v) for v in values],
            }
            for col, values in features.items()
        ],
    }
    url = predictor_url(name, namespace=namespace)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"KServe inference failed ({exc.code}): {detail}") from exc

    outputs = body.get("outputs") or []
    if not outputs:
        raise RuntimeError(f"KServe response missing outputs: {body!r}")
    data = outputs[0].get("data")
    if data is None:
        raise RuntimeError(f"KServe response missing output data: {body!r}")
    if len(data) != n_rows:
        raise RuntimeError(
            f"Expected {n_rows} prediction(s), got {len(data)} from {name!r}"
        )
    return [float(v) for v in data]


def invoke_tensor_predictor(
    name: str,
    input_tensors: Mapping[str, Mapping[str, Any]],
    *,
    namespace: str | None = None,
    timeout_seconds: int = 60,
) -> dict[str, dict[str, Any]]:
    """Run inference against any predictor speaking raw KServe V2 tensors.

    Unlike :func:`invoke_predictor` (flat per-column tabular features, one
    value per row), this passes arbitrary named V2 tensors straight through —
    for predictors whose inputs aren't tabular (e.g. a custom container with
    its own tensor shapes, such as AIHydroPoints's time-series predictor,
    which takes one tensor per forcing variable shaped ``[nlocations, ntimes]``
    plus a shared ``times`` tensor). ``input_tensors`` maps a tensor name to
    its V2 tensor body (``{"shape": [...], "datatype": ..., "data": [...]}``).
    Returns the response ``outputs`` keyed by tensor name, unparsed — the
    caller reshapes ``data`` using ``shape``.
    """
    if not input_tensors:
        raise ValueError("input_tensors must not be empty")

    payload = {
        "inputs": [{"name": tname, **tensor} for tname, tensor in input_tensors.items()],
    }
    url = predictor_url(name, namespace=namespace)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"KServe inference failed ({exc.code}): {detail}") from exc

    outputs = body.get("outputs") or []
    if not outputs:
        raise RuntimeError(f"KServe response missing outputs: {body!r}")
    return {o["name"]: o for o in outputs}


def wait_for_inference_service_ready(
    name: str,
    *,
    namespace: str | None = None,
    timeout_seconds: int = 600,
    poll_interval: int = 15,
) -> None:
    """Wait until KServe reports the InferenceService Ready condition."""
    ns = namespace or runtime_namespace()
    api = _custom_objects_api()
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        obj = api.get_namespaced_custom_object(
            KSERVE_GROUP, KSERVE_VERSION, ns, KSERVE_PLURAL, name
        )
        status = obj.get("status") or {}
        for cond in status.get("conditions") or []:
            if cond.get("type") == "Ready" and cond.get("status") == "True":
                log.info("InferenceService %s is Ready", name)
                return
        fail = (status.get("modelStatus") or {}).get("lastFailureInfo") or {}
        if fail:
            log.warning(
                "InferenceService %s not Ready: %s — %s",
                name,
                fail.get("reason"),
                (fail.get("message") or "")[:300],
            )
        else:
            log.info("Waiting for InferenceService %s…", name)
        time.sleep(poll_interval)
    raise TimeoutError(f"InferenceService {name!r} not Ready after {timeout_seconds}s in {ns!r}")
