"""Ray Job Submission API helpers — submit_ray_job / wait_for_ray_job.

Chosen deliberately over the two alternatives:
- **Raw Ray Client** (`ray.init(address="ray://...")`) is discouraged by Ray's
  own docs for anything beyond quick interactive use — the entire session
  dies if that connection drops even briefly, with no resume. Not something to
  pin an unattended Airflow DAG run on.
- **The KubeRay `RayJob` CRD** is more machinery than needed — it requires
  speaking the Kubernetes API purely to reach a HTTP endpoint the cluster
  already exposes directly.

One HTTP call to the cluster's dashboard is enough: the submitted entrypoint
runs as a driver *on the cluster itself*, decoupled from whatever submitted
it — the calling Airflow task can be rescheduled without killing the job.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from pathlib import Path
from urllib.parse import urlsplit

from ray.job_submission import JobStatus, JobSubmissionClient

log = logging.getLogger(__name__)

# Cross-namespace: from both ``platform`` and ``pr-*``.
RAY_DASHBOARD_URL = os.environ.get(
    "RAY_DASHBOARD_URL",
    "http://platform-ray-head-svc.platform.svc.cluster.local:8265",
)

_TERMINAL_STATUSES = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}


def mlops_repo_root() -> Path:
    """Airflow's git-synced ai-platform-mlops checkout.

    The source `py_modules` runtime_envs upload code from, so a submitted Ray
    job runs the exact same code already on Airflow's own PYTHONPATH — no
    ray-worker image rebuild needed to pick up a logic change.
    """
    return Path(os.environ.get("AIRFLOW_MLOPS_REPO", "/opt/airflow/dags/repo"))


def _mlflow_uri_for_ray_job() -> str:
    """Turn a short MLflow hostname like ``mlflow:5000`` into its IP.

    Ray jobs always run in ``platform``, so a short name would point to the
    wrong MLflow. The IP fixes that and is allowed either way.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    parsed = urlsplit(uri)
    host = parsed.hostname or "mlflow"
    if "." in host:
        return uri
    ip = socket.gethostbyname(host)
    port = f":{parsed.port}" if parsed.port else ""
    return f"{parsed.scheme}://{ip}{port}"


def ray_job_env_vars() -> dict[str, str]:
    """MLflow + S3 env vars for a submitted Ray job.

    Ray jobs start with a blank env, so these must be passed explicitly or
    the job can't reach MLflow or S3.
    """
    keys = (
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "MLFLOW_S3_ENDPOINT_URL",
        "S3_ENDPOINT",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    )
    env: dict[str, str] = {
        "MLFLOW_S3_IGNORE_TLS": "true",
        "MLFLOW_BOTO_CLIENT_ADDRESSING_STYLE": "path",
        "AWS_DEFAULT_REGION": "eu-west-1",
        "RAY_DEDUP_LOGS": "0",
    }
    for key in keys:
        val = os.environ.get(key)
        if val:
            env[key] = val
    env["MLFLOW_TRACKING_URI"] = _mlflow_uri_for_ray_job()
    env.setdefault("MLFLOW_S3_ENDPOINT_URL", env.get("S3_ENDPOINT", "https://s3.deltares.nl"))
    return env


def submit_ray_job(
    entrypoint: str, *, runtime_env: dict | None = None, submission_id: str | None = None
) -> str:
    """Submit `entrypoint` (a shell command) to run as a driver on the Ray cluster.

    Returns the job_id immediately; the submitting process does not stay
    connected — use `wait_for_ray_job` to poll for completion.

    `submission_id` makes submission idempotent: resubmitting the same id
    raises instead of starting a second job.
    """
    client = JobSubmissionClient(RAY_DASHBOARD_URL)
    job_id = client.submit_job(
        entrypoint=entrypoint, runtime_env=runtime_env, submission_id=submission_id
    )
    log.info("submitted Ray job %s: %s", job_id, entrypoint)
    return job_id


def stop_ray_job(job_id: str) -> None:
    """Request the cluster stop `job_id`."""
    client = JobSubmissionClient(RAY_DASHBOARD_URL)
    client.stop_job(job_id)
    log.info("stopped Ray job %s", job_id)


def wait_for_ray_job(job_id: str, *, poll_interval: int = 15, timeout_seconds: int = 3600) -> None:
    """Poll `job_id` until it reaches a terminal status, streaming new log output
    into the caller's logger as it arrives.

    Raises `RuntimeError` on FAILED/STOPPED (so the calling PythonOperator fails
    the Airflow task naturally) and `TimeoutError` if it never finishes. Safe to
    re-run after the submitting process itself gets rescheduled — the job keeps
    running on the cluster independent of the submitter.
    """
    client = JobSubmissionClient(RAY_DASHBOARD_URL)
    deadline = time.monotonic() + timeout_seconds
    seen = 0
    status = client.get_job_status(job_id)

    while status not in _TERMINAL_STATUSES:
        logs = client.get_job_logs(job_id)
        if len(logs) > seen:
            for line in logs[seen:].splitlines():
                log.info("[ray %s] %s", job_id, line)
            seen = len(logs)

        if time.monotonic() > deadline:
            stop_ray_job(job_id)
            raise TimeoutError(f"Ray job {job_id} still {status} after {timeout_seconds}s")
        time.sleep(poll_interval)
        status = client.get_job_status(job_id)

    # Flush any log lines written between the last poll and the terminal status.
    logs = client.get_job_logs(job_id)
    if len(logs) > seen:
        for line in logs[seen:].splitlines():
            log.info("[ray %s] %s", job_id, line)

    if status != JobStatus.SUCCEEDED:
        raise RuntimeError(f"Ray job {job_id} ended with status {status}")
    log.info("Ray job %s succeeded", job_id)
