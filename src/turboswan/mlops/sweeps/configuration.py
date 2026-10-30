"""Validated sweep TOML configuration and parameter-space definitions."""

from __future__ import annotations

import math
import pathlib
import random
import tomllib
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator, model_validator


class Categorical(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: list

    @model_validator(mode="after")
    def _non_empty(self) -> Categorical:
        if not self.values:
            raise ValueError("categorical search space entry has no values")
        return self


class Distribution(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["int", "float", "logfloat"]
    low: float
    high: float

    @model_validator(mode="after")
    def _ordered_range(self) -> Distribution:
        if self.low >= self.high:
            raise ValueError(f"low {self.low} must be < high {self.high}")
        if self.kind == "logfloat" and self.low <= 0:
            raise ValueError(f"logfloat low {self.low} must be > 0")
        return self


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_config_toml: str
    selection_metric: str = "val_loss"
    direction: Literal["minimize", "maximize"] = "minimize"
    include_baseline: bool = True


class PrunerConfig(BaseModel):
    """
    ``name`` must match an ``optuna.pruners`` name (e.g., 'MedianPruner').
    Extra fields are assumed to be parameters of the specified pruner.
    """
    
    model_config = ConfigDict(extra="allow")
    
    name: str
    
    @field_validator("name")
    @classmethod
    def _is_optuna_pruner(cls, name: str) -> str:
        from optuna import pruners  # lazy: only configs with a pruner need optuna
        
        valid = [n for n in pruners.__all__ if n.endswith("Pruner") and n != "BasePruner"]
        if name not in valid:
            raise ValueError(f"{name!r} is not an optuna pruner; choose from {valid}")
        return name


class StrategyConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Literal["random", "optuna"] = "random"
    max_configs: PositiveInt
    seed: int = 0
    max_concurrent: PositiveInt = 1
    pruner: PrunerConfig | None = None

    @model_validator(mode="after")
    def _pruner_needs_optuna(self) -> StrategyConfig:
        if self.pruner is not None and self.name != "optuna":
            raise ValueError(f"a pruner needs strategy name 'optuna', not {self.name!r}")
        return self


def parse_search_space(raw: dict) -> dict[str, Categorical | Distribution]:
    """Parse list choices and validated numeric distributions from TOML input."""
    return {
        key: Categorical(values=spec) if isinstance(spec, list) else Distribution.model_validate(spec)
        for key, spec in raw.items()
    }


def load_sweep(
    sweep_config_path: pathlib.Path,
) -> tuple[SweepConfig, StrategyConfig, dict[str, Categorical | Distribution]]:
    """Load the sweep settings, strategy settings, and search space from TOML."""
    with sweep_config_path.open("rb") as sweep_file:
        raw = tomllib.load(sweep_file)
    return (
        SweepConfig.model_validate(raw["sweep"]),
        StrategyConfig.model_validate(raw["strategy"]),
        parse_search_space(raw.get("search_space", {})),
    )


def sample_one(spec: Categorical | Distribution, rng: random.Random):
    """Draw one configured categorical or numeric parameter value."""
    if isinstance(spec, Categorical):
        return rng.choice(spec.values)
    if spec.kind == "int":
        return rng.randint(int(spec.low), int(spec.high))
    if spec.kind == "logfloat":
        return math.exp(rng.uniform(math.log(spec.low), math.log(spec.high)))
    return rng.uniform(spec.low, spec.high)


def sample_configs(
    search_space: dict[str, Categorical | Distribution],
    max_configs: int,
    seed: int,
) -> list[dict]:
    """Draw ``max_configs`` random dotted-key parameter override dictionaries."""
    rng = random.Random(seed)
    return [
        {key: sample_one(spec, rng) for key, spec in search_space.items()}
        for _ in range(max_configs)
    ]

__all__ = [
    "Categorical",
    "Distribution",
    "PrunerConfig",
    "StrategyConfig",
    "SweepConfig",
    "load_sweep",
    "parse_search_space",
    "sample_configs",
    "sample_one",
]