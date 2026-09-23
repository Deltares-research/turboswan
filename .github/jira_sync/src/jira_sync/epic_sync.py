"""Entry point for full bidirectional Jira and GitHub epic reconciliation."""

from __future__ import annotations

from . import github_to_jira
from . import jira_to_github


def main() -> None:
    jira_to_github.sync_all()
    github_to_jira.sync_all()


if __name__ == "__main__":
    main()
