"""Operational capabilities added around the researcher-owned TurboSWAN code.

The ``core`` package and ``training`` module retain the scientific workflow.
This package holds the optional MLflow tracking and parameter-sweep mechanics
used by ``training_mlflow`` and ``training_sweep``, without changing training
behavior.
"""