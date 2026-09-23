"""Shared pytest fixtures for the jira_sync package tests."""

import os

import pytest


# Modules read this setting during import, before autouse fixtures run.
os.environ.setdefault("JIRA_EPIC_KEY", "MLPS-118")


@pytest.fixture(autouse=True)
def required_env(monkeypatch):
    """Provide dummy credentials so modules can be imported/used in isolation."""
    monkeypatch.setenv("JIRA_BASE_URL", "https://example.atlassian.net")
    monkeypatch.setenv("JIRA_EMAIL", "bot@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "dummy-token")
    monkeypatch.setenv("JIRA_EPIC_KEY", "MLPS-118")
    monkeypatch.setenv("GITHUB_TOKEN", "dummy-gh-token")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
