"""GitHub-side reconciliation for the Jira epic sync.

Each full sync scans GitHub issues and pull requests that reference the epic,
plus GitHub items already linked from Jira. It creates missing Jira issues,
keeps descriptions in sync, and mirrors open/closed state. Links are stored
in description footers rather than labels or title markers.

Required env vars: JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, GITHUB_TOKEN,
GITHUB_REPOSITORY (set automatically by GitHub Actions).

Optional env vars:
    JIRA_EPIC_KEY          required Jira epic key to synchronize
"""

from __future__ import annotations

import os

from . import github_client as gh
from . import jira_client as jira
from .linking import (
    build_jira_description,
    ensure_jira_description_postfix,
    parse_gh_number,
    sync_footer,
)

JIRA_EPIC_KEY = os.environ["JIRA_EPIC_KEY"]
JIRA_JQL = f'parent = "{JIRA_EPIC_KEY}"'
JIRA_PROJECT_KEY = JIRA_EPIC_KEY.split("-")[0]
JIRA_ISSUE_TYPE = "Task"
JIRA_DONE_STATUS = "Done"
JIRA_REOPEN_STATUS = "To Do"


def find_linked_jira_key(number: int) -> str | None:
    """Find the Jira issue whose sync footer links to a GitHub item."""
    for issue in jira.search(JIRA_JQL, fields=["description"]):
        description = jira.adf_to_text(issue["fields"].get("description"))
        if parse_gh_number(description) == number:
            return issue["key"]
    return None


def sync_item(item: dict) -> None:
    """Reconcile a single GitHub issue/PR with its linked Jira issue."""
    number = item["number"]
    title = item.get("title", "")
    body = item.get("body") or ""

    jira_key = find_linked_jira_key(number)
    jira_issue = jira.get_issue(jira_key) if jira_key else None

    if jira_key is not None and jira_issue is None:
        # The Jira issue behind this marker was deleted/inaccessible; forget
        # the stale link and re-create below instead of giving up.
        print(f"#{number}: linked Jira issue {jira_key} no longer exists, re-linking")
        jira_key = None

    if jira_key is None:
        # Only auto-create a Jira issue if the unlinked item mentions the epic.
        if JIRA_EPIC_KEY.lower() not in f"{title}\n{body}".lower():
            print(f"#{number} does not reference {JIRA_EPIC_KEY}, skipping")
            return
        gh_url = item["html_url"]
        jira_key = jira.create_issue(
            project_key=JIRA_PROJECT_KEY,
            issue_type=JIRA_ISSUE_TYPE,
            summary=title,
            description=build_jira_description(gh_url, body),
            epic_key=JIRA_EPIC_KEY,
        )
        if jira_key is None:
            print(f"#{number}: failed to create a linked Jira issue, skipping")
            return
        print(f"Created {jira_key} for #{number}")
        return

    # Item was already linked; ensure the description has the tracking
    # footer (without clobbering any content added directly in Jira) and
    # mirror the open/closed state.
    current_description = jira.adf_to_text(jira_issue["fields"].get("description"))
    updated_description = ensure_jira_description_postfix(
        current_description, item["html_url"]
    )
    if updated_description is not None:
        jira.update_description(jira_key, updated_description)
        print(f"Added GitHub sync footer to {jira_key} description")

    jira_done = jira_issue["fields"]["status"]["statusCategory"]["key"] == "done"
    gh_open = item.get("state", "open") == "open"

    if gh_open and jira_done:
        jira.transition_issue(jira_key, JIRA_REOPEN_STATUS)
        print(f"Transitioned {jira_key} to {JIRA_REOPEN_STATUS} (#{number} is open)")
    elif not gh_open and not jira_done:
        jira.transition_issue(jira_key, JIRA_DONE_STATUS)
        print(f"Transitioned {jira_key} to {JIRA_DONE_STATUS} (#{number} is closed)")
    else:
        print(f"{jira_key}: already in sync with #{number}")


def sync_all() -> None:
    """Reconcile every GitHub issue/PR referencing the epic.

    Discovers items two ways so both sides of the sync stay authoritative:
    a GitHub text search for the epic key (catches new/unlinked items), and
    the same Jira epic query jira_to_github uses (catches items whose
    GitHub text no longer mentions the epic but are still linked).
    """
    repo = os.environ["GITHUB_REPOSITORY"]
    epic_footer = sync_footer(
        f"{os.environ['JIRA_BASE_URL'].rstrip('/')}/browse/{JIRA_EPIC_KEY}"
    )
    query = f'repo:{repo} "{JIRA_EPIC_KEY}" in:body,title'
    items_by_number: dict[int, dict] = {}

    for item in gh.search_issues(query):
        body = item.get("body") or ""
        if body.rstrip().endswith(epic_footer):
            continue
        items_by_number[item["number"]] = item

    for jira_issue in jira.search(JIRA_JQL, fields=["description"]):
        description = jira.adf_to_text(jira_issue["fields"].get("description"))
        number = parse_gh_number(description)
        if number is None or number in items_by_number:
            continue
        gh_item = gh.get_issue(number)
        if gh_item is not None:
            items_by_number[number] = gh_item

    print(
        f"Found {len(items_by_number)} GitHub item(s) to reconcile against {JIRA_EPIC_KEY}"
    )
    for item in items_by_number.values():
        sync_item(item)
