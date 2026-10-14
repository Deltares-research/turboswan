import importlib
from unittest.mock import patch

import pytest
from jira_sync import epic_sync


def test_directional_modules_are_importable():
    # Arrange
    from jira_sync import jira_to_github

    # Act
    directional_modules = (jira_to_github,)

    # Assert
    assert all(module is not None for module in directional_modules)


def test_epic_key_is_required(monkeypatch):
    # Arrange
    monkeypatch.delenv("JIRA_EPIC_KEY", raising=False)

    # Act and assert
    with pytest.raises(KeyError):
        importlib.reload(epic_sync.jira_to_github)

    # Assert
    monkeypatch.setenv("JIRA_EPIC_KEY", "MLPS-118")
    importlib.reload(epic_sync.jira_to_github)


def test_main_reconciles_jira_to_github():
    # Arrange
    with (
        patch(
            "jira_sync.epic_sync.jira_client.get_issue",
            return_value={"key": "MLPS-118"},
        ) as get_issue,
        patch("jira_sync.epic_sync.jira_to_github.sync_all") as sync_jira_to_github,
    ):
        # Act
        epic_sync.main()

    # Assert
    get_issue.assert_called_once_with("MLPS-118")
    sync_jira_to_github.assert_called_once_with()


def test_main_aborts_when_epic_is_inaccessible():
    # Arrange
    with (
        patch("jira_sync.epic_sync.jira_client.get_issue", return_value=None),
        patch("jira_sync.epic_sync.jira_to_github.sync_all") as sync_jira_to_github,
        pytest.raises(RuntimeError, match="Cannot access Jira epic MLPS-118"),
    ):
        # Act
        epic_sync.main()

    # Assert
    sync_jira_to_github.assert_not_called()
