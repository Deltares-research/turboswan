"""Jira reconciliation helpers for bidirectional sync.

Each full sync finds every Jira issue under the tracked epic and mirrors it
onto GitHub: issues not yet linked to a GitHub issue/PR get a new
GitHub issue created, with a sync footer added to the Jira description to
record the link; already-linked issues have their description kept in sync with the
Jira summary/description, their comments mirrored, and their Jira status
mirrored back onto GitHub (open <-> closed).

Required env vars: JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, GITHUB_TOKEN,
GITHUB_REPOSITORY.

Optional env vars:
    JIRA_EPIC_KEY   required Jira epic key to synchronize
"""

from __future__ import annotations

import os
import re

from . import github_client as gh
from . import jira_client as jira
from .linking import (
    append_sync_footer,
    build_jira_description,
    ensure_jira_description_postfix,
    parse_gh_number,
    remove_sync_footer,
    sync_footer,
)

JIRA_EPIC_KEY = os.environ["JIRA_EPIC_KEY"]
JIRA_JQL = f'parent = "{JIRA_EPIC_KEY}"'

JIRA_COMMENT_ID_RE = re.compile(r"<!-- jira-comment-id:(\S+) -->")

# Marker prefixes used to recognize comments that were themselves produced by
# this sync (either direction), so they're never re-mirrored and can't loop.
JIRA_TO_GH_COMMENT_PREFIX = "_Synced from Jira comment"
GH_TO_JIRA_COMMENT_PREFIX = "Synced from GitHub comment"


def build_synced_body(key: str, description: str) -> str:
    jira_url = f"{os.environ['JIRA_BASE_URL'].rstrip('/')}/browse/{key}"
    return append_sync_footer(description, jira_url)


def create_and_link_gh_issue(summary: str, body: str) -> int | None:
    return gh.create_issue(title=summary, body=body)


def build_epic_body(description: str) -> str:
    jira_url = f"{os.environ['JIRA_BASE_URL'].rstrip('/')}/browse/{JIRA_EPIC_KEY}"
    return append_sync_footer(description, jira_url)


def ensure_epic_issue() -> dict | None:
    epic = jira.get_issue(JIRA_EPIC_KEY)
    if epic is None:
        print(f"{JIRA_EPIC_KEY}: Jira epic not found, skipping GitHub parent sync")
        return None

    summary = epic["fields"]["summary"]
    description = jira.adf_to_text(epic["fields"].get("description"))
    jira_url = f"{os.environ['JIRA_BASE_URL'].rstrip('/')}/browse/{JIRA_EPIC_KEY}"
    updated_description = ensure_jira_description_postfix(description, jira_url)
    if updated_description is not None:
        jira.update_description(JIRA_EPIC_KEY, updated_description)
        description = updated_description
    expected_body = build_epic_body(description)
    footer = sync_footer(jira_url)
    repo = os.environ["GITHUB_REPOSITORY"]

    for item in gh.search_issues(f'repo:{repo} "{JIRA_EPIC_KEY}" in:body'):
        body = item.get("body") or ""
        if not body.rstrip().endswith(footer):
            continue
        if item.get("title") != summary or item.get("body") != expected_body:
            gh.update_issue(item["number"], title=summary, body=expected_body)
        return gh.get_issue(item["number"]) or item

    number = gh.create_issue(summary, expected_body)
    if number is None:
        print(f"{JIRA_EPIC_KEY}: failed to create GitHub parent issue")
        return None
    print(f"{JIRA_EPIC_KEY}: created GitHub parent #{number}")
    return gh.get_issue(number)


def link_to_epic(parent_issue: dict | None, child_issue: dict) -> None:
    if parent_issue is None or parent_issue["number"] == child_issue["number"]:
        return
    child_id = child_issue.get("id")
    if child_id is None:
        print(
            f"#{child_issue['number']}: missing GitHub issue id, skipping parent link"
        )
        return
    parent_number = parent_issue["number"]
    if child_id not in gh.list_sub_issue_ids(parent_number):
        gh.add_sub_issue(parent_number, child_id)


def sync_comments(key: str, number: int) -> None:
    already_mirrored: set[str] = set()
    for gh_comment in gh.list_comments(number):
        match = JIRA_COMMENT_ID_RE.search(gh_comment.get("body", ""))
        if match:
            already_mirrored.add(match.group(1))

    for comment in jira.list_comments(key):
        comment_id = comment["id"]
        if comment_id in already_mirrored:
            continue
        text = jira.adf_to_text(comment.get("body"))
        if text.startswith(GH_TO_JIRA_COMMENT_PREFIX):
            # This comment originated on GitHub; no need to mirror it back.
            continue
        author = comment.get("author", {}).get("displayName", "Jira user")
        body = (
            f"{JIRA_TO_GH_COMMENT_PREFIX} by {author}:_\n\n{text}\n\n"
            f"<!-- jira-comment-id:{comment_id} -->"
        )
        gh.add_comment(number, body)
        print(f"{key}: mirrored Jira comment {comment_id} to GitHub #{number}")


def sync_all() -> None:
    issues = jira.search(JIRA_JQL, fields=["status", "summary", "description"])
    print(f"Found {len(issues)} Jira issue(s) under {JIRA_EPIC_KEY}")
    parent_issue = ensure_epic_issue()

    for issue in issues:
        key = issue["key"]
        description = jira.adf_to_text(issue["fields"].get("description"))
        number = parse_gh_number(description)
        had_tracking_link = number is not None
        gh_item = gh.get_issue(number) if number is not None else None

        if number is not None and gh_item is None:
            print(f"{key}: linked GitHub #{number} not found, recreating")
            description = remove_sync_footer(description)
            number = None

        if number is not None:
            gh_url = (
                f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/issues/{number}"
            )
            updated_description = ensure_jira_description_postfix(description, gh_url)
            if updated_description is not None:
                jira.update_description(key, updated_description)
                description = updated_description

        expected_body = build_synced_body(key, description)

        if number is None:
            number = create_and_link_gh_issue(issue["fields"]["summary"], expected_body)
            if number is None:
                print(f"{key}: failed to create a linked GitHub issue, skipping")
                continue
            gh_url = (
                f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/issues/{number}"
            )
            if had_tracking_link:
                updated_description = build_jira_description(gh_url, description)
            else:
                updated_description = ensure_jira_description_postfix(
                    description, gh_url
                )
            if updated_description is not None:
                jira.update_description(key, updated_description)
            print(f"{key}: created and linked GitHub #{number}")

        status_category = issue["fields"]["status"]["statusCategory"]["key"]
        jira_done = status_category == "done"

        if gh_item is None:
            gh_item = gh.get_issue(number)
        if gh_item is None:
            print(f"{key}: linked GitHub #{number} not found, skipping")
            continue

        link_to_epic(parent_issue, gh_item)

        if gh_item.get("body") != expected_body:
            gh.update_issue(number, body=expected_body)
            print(f"{key}: updated GitHub #{number} description from Jira")

        sync_comments(key, number)

        gh_open = gh_item["state"] == "open"

        if jira_done and gh_open:
            gh.add_comment(
                number, f"Closing automatically: {key} was marked Done in Jira."
            )
            gh.set_state(number, "closed")
            print(f"Closed #{number} ({key} is Done)")
        elif not jira_done and not gh_open:
            gh.add_comment(
                number, f"Reopening automatically: {key} is no longer Done in Jira."
            )
            gh.set_state(number, "open")
            print(f"Reopened #{number} ({key} is not Done)")
        else:
            print(
                f"{key}: #{number} already in sync (jira_done={jira_done}, gh_open={gh_open})"
            )
