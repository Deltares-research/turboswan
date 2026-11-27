"""MLflow GenAI evaluate helpers (datasets, stub scorers)."""

from __future__ import annotations

import os
from typing import Any

# mlflow.genai.evaluate defaults to 10 predict + 10 scorer threads. Under Airflow
# that exhausts the MLflow client's urllib3 pool (size 10) and the task stalls.
# Serialising costs nothing measurable when the work is network-bound anyway.
AIRFLOW_GENAI_EVAL_ENV: dict[str, str] = {
    "MLFLOW_GENAI_EVAL_MAX_WORKERS": "1",
    "MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS": "1",
    "MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION": "True",
}


def pin_airflow_genai_eval_concurrency() -> dict[str, str]:
    """Pin GenAI evaluate to single-threaded mode; return the effective env values."""
    for key, value in AIRFLOW_GENAI_EVAL_ENV.items():
        os.environ.setdefault(key, value)
    return {key: os.environ[key] for key in AIRFLOW_GENAI_EVAL_ENV}


def ensure_genai_dataset(
    name: str,
    experiment_id: str,
    records: list[dict[str, Any]],
    *,
    tags: dict[str, str] | None = None,
) -> Any:
    """Return an existing GenAI dataset by name, or create one and merge records."""
    from mlflow.genai.datasets import create_dataset, get_dataset

    try:
        dataset = get_dataset(name=name)
        print(f"Reusing dataset {name!r} ({dataset.dataset_id})")
        return dataset
    except Exception:
        pass

    dataset = create_dataset(name=name, experiment_id=experiment_id, tags=tags or {})
    dataset = dataset.merge_records(records)
    print(f"Created dataset {name!r} with {len(records)} records")
    return dataset


def constant_stub_scorer(name: str, value: float = 0.5) -> Any:
    """Scorer that always returns a fixed value — for toy/smoke GenAI evaluate runs."""
    from mlflow.genai.scorers import scorer

    def _fn(outputs: str) -> float:
        return value

    _fn.__name__ = name
    _fn.__qualname__ = name
    return scorer(name=name, pass_if=lambda v: v >= 0.0)(_fn)


def json_result_scorer(name: str, results_by_qid: dict[str, dict]) -> Any:
    """Scorer that returns a pre-computed metric from S3 evaluation JSON (by question_id)."""
    from mlflow.genai.scorers import scorer

    def _fn(outputs: str, expectations: dict) -> float:
        qid = str(expectations.get("question_id", ""))
        if qid not in results_by_qid:
            raise KeyError(f"No demo result for question_id={qid!r}")
        data = results_by_qid[qid]
        if name == "ragas_score":
            return float(data["ragas_score"])
        return float(data["metrics"][name]["score"])

    _fn.__name__ = name
    _fn.__qualname__ = name
    return scorer(name=name, pass_if=lambda v: v >= 0.0)(_fn)
