import importlib
from unittest.mock import patch

import pytest

from jira_sync import epic_sync


def test_directional_modules_are_importable():
    import jira_sync.github_to_jira as github_to_jira
    import jira_sync.jira_to_github as jira_to_github

    assert github_to_jira is not None
    assert jira_to_github is not None


def test_epic_key_is_required(monkeypatch):
    monkeypatch.delenv("JIRA_EPIC_KEY", raising=False)

    with pytest.raises(KeyError):
        importlib.reload(epic_sync.github_to_jira)

    monkeypatch.setenv("JIRA_EPIC_KEY", "MLPS-118")
    importlib.reload(epic_sync.github_to_jira)


def test_main_reconciles_both_directions():
    with patch(
        "jira_sync.epic_sync.jira_to_github.sync_all"
    ) as sync_jira_to_github, patch(
        "jira_sync.epic_sync.github_to_jira.sync_all"
    ) as sync_github_to_jira:
        epic_sync.main()

    sync_jira_to_github.assert_called_once_with()
    sync_github_to_jira.assert_called_once_with()
