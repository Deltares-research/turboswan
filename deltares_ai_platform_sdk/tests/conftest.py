"""Shared fixtures for the SDK tests."""

from pathlib import Path

import pytest


@pytest.fixture
def optuna_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Point Optuna at a throwaway sqlite database and return its URL."""
    url = f"sqlite:///{(tmp_path / 'studies.db').as_posix()}"
    monkeypatch.setenv("OPTUNA_STORAGE", url)
    return url
