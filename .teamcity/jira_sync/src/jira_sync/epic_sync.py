"""Entry point for Jira-to-GitHub epic synchronization."""

from __future__ import annotations

import os

from jira_sync import jira_client, jira_to_github


def main() -> None:
    epic_key = os.environ["JIRA_EPIC_KEY"]
    if jira_client.get_issue(epic_key) is None:
        raise RuntimeError(
            f"Cannot access Jira epic {epic_key} at {os.environ['JIRA_BASE_URL']} "
            f"using JIRA_EMAIL={os.environ['JIRA_EMAIL']}. "
            "Check the CI Jira site, token owner, token scopes, Browse Projects "
            "permission, and issue security. Sync aborted before making changes."
        )
    jira_to_github.sync_all()


if __name__ == "__main__":
    main()
