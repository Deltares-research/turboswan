from unittest.mock import patch

from jira_sync import github_to_jira as module


def github_item(number=1, title="Fix things", body="", state="open"):
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "html_url": f"https://github.com/owner/repo/issues/{number}",
    }


def test_github_to_jira_find_linked_jira_key_returns_match():
    with patch(
        "jira_sync.github_to_jira.jira.search",
        return_value=[
            {
                "key": "MLPS-42",
                "fields": {
                    "description": module.jira.text_to_adf(
                        "Automatically synced at: https://github.com/owner/repo/issues/1"
                    )
                },
            }
        ],
    ) as search:
        assert module.find_linked_jira_key(1) == "MLPS-42"
    search.assert_called_once_with(module.JIRA_JQL, fields=["description"])


def test_github_to_jira_find_linked_jira_key_returns_none_when_absent():
    with patch("jira_sync.github_to_jira.jira.search", return_value=[]):
        assert module.find_linked_jira_key(1) is None


def test_github_to_jira_find_linked_jira_key_verifies_exact_number_match():
    # Only an exact GH URL match should count, even if another issue's
    # description happens to link to a similar-looking number (e.g. #10).
    with patch(
        "jira_sync.github_to_jira.jira.search",
        return_value=[
            {
                "key": "MLPS-99",
                "fields": {
                    "description": module.jira.text_to_adf(
                        "Automatically synced at: https://github.com/owner/repo/issues/10"
                    )
                },
            }
        ],
    ):
        assert module.find_linked_jira_key(1) is None


def test_github_to_jira_skips_when_epic_key_not_mentioned():
    item = github_item(body="unrelated work")

    with patch("jira_sync.github_to_jira.jira.search", return_value=[]), patch(
        "jira_sync.github_to_jira.jira.create_issue"
    ) as create_issue:
        module.sync_item(item)

    create_issue.assert_not_called()


def test_github_to_jira_creates_jira_issue_for_unlinked_item_referencing_epic():
    item = github_item(number=7, body="Part of MLPS-118")

    with patch("jira_sync.github_to_jira.jira.search", return_value=[]), patch(
        "jira_sync.github_to_jira.jira.create_issue", return_value="MLPS-101"
    ) as create_issue:
        module.sync_item(item)

    create_issue.assert_called_once()
    assert create_issue.call_args.kwargs["summary"] == "Fix things"
    assert create_issue.call_args.kwargs["description"] == (
        "Part of MLPS-118\n\n----------------------------------------\nAutomatically synced at: https://github.com/owner/repo/issues/7"
    )
    assert create_issue.call_args.kwargs["issue_type"] == "Task"


def make_jira_issue(description="", status_category="new"):
    return {
        "fields": {
            "description": module.jira.text_to_adf(description)
            if description
            else None,
            "status": {"statusCategory": {"key": status_category}},
        }
    }


def test_github_to_jira_syncs_item_found_via_jira_description_link():
    # Simulates an issue created by jira_reconciliation (description ending
    # with "Automatically synced at: <url>"); the footer carries the link.
    item = github_item(body="Part of MLPS-118 updated")
    tracked_description = "human-written summary\n\n----------------------------------------\nAutomatically synced at: https://github.com/owner/repo/issues/1"
    jira_issue = make_jira_issue(description=tracked_description, status_category="new")

    with patch(
        "jira_sync.github_to_jira.jira.search",
        return_value=[
            {
                "key": "MLPS-110",
                "fields": {"description": module.jira.text_to_adf(tracked_description)},
            }
        ],
    ), patch("jira_sync.github_to_jira.jira.create_issue") as create_issue, patch(
        "jira_sync.github_to_jira.jira.get_issue", return_value=jira_issue
    ), patch(
        "jira_sync.github_to_jira.jira.update_description"
    ) as update_description, patch(
        "jira_sync.github_to_jira.jira.transition_issue"
    ), patch("jira_sync.github_to_jira.gh.add_comment") as add_comment:
        module.sync_item(item)

    create_issue.assert_not_called()
    add_comment.assert_not_called()
    # The description already has the tracking footer, so nothing changes.
    update_description.assert_not_called()


def test_github_to_jira_transitions_to_done_when_closed_and_jira_not_done():
    item = github_item(body="Part of MLPS-118", state="closed")
    already_prefixed = module.jira.text_to_adf(
        "Automatically synced at: https://github.com/owner/repo/issues/1"
    )
    jira_issue = {
        "fields": {
            "description": already_prefixed,
            "status": {"statusCategory": {"key": "new"}},
        }
    }

    with patch(
        "jira_sync.github_to_jira.jira.search",
        return_value=[{"key": "MLPS-55", "fields": {"description": already_prefixed}}],
    ), patch("jira_sync.github_to_jira.jira.get_issue", return_value=jira_issue), patch(
        "jira_sync.github_to_jira.jira.update_description"
    ), patch("jira_sync.github_to_jira.jira.transition_issue") as transition_issue:
        module.sync_item(item)

    transition_issue.assert_called_once_with("MLPS-55", module.JIRA_DONE_STATUS)


def test_github_to_jira_transitions_to_reopen_when_open_and_jira_done():
    item = github_item(body="Part of MLPS-118", state="open")
    already_prefixed = module.jira.text_to_adf(
        "Automatically synced at: https://github.com/owner/repo/issues/1"
    )
    jira_issue = {
        "fields": {
            "description": already_prefixed,
            "status": {"statusCategory": {"key": "done"}},
        }
    }

    with patch(
        "jira_sync.github_to_jira.jira.search",
        return_value=[{"key": "MLPS-55", "fields": {"description": already_prefixed}}],
    ), patch("jira_sync.github_to_jira.jira.get_issue", return_value=jira_issue), patch(
        "jira_sync.github_to_jira.jira.update_description"
    ), patch("jira_sync.github_to_jira.jira.transition_issue") as transition_issue:
        module.sync_item(item)

    transition_issue.assert_called_once_with("MLPS-55", module.JIRA_REOPEN_STATUS)


def test_github_to_jira_no_transition_when_already_in_sync():
    item = github_item(body="Part of MLPS-118", state="open")
    already_prefixed = module.jira.text_to_adf(
        "Automatically synced at: https://github.com/owner/repo/issues/1"
    )
    jira_issue = {
        "fields": {
            "description": already_prefixed,
            "status": {"statusCategory": {"key": "new"}},
        }
    }

    with patch(
        "jira_sync.github_to_jira.jira.search",
        return_value=[{"key": "MLPS-55", "fields": {"description": already_prefixed}}],
    ), patch("jira_sync.github_to_jira.jira.get_issue", return_value=jira_issue), patch(
        "jira_sync.github_to_jira.jira.update_description"
    ), patch("jira_sync.github_to_jira.jira.transition_issue") as transition_issue:
        module.sync_item(item)

    transition_issue.assert_not_called()


def test_github_to_jira_syncs_already_linked_item_even_without_epic_mention():
    # Issues created by jira_reconciliation have titles/bodies referencing
    # their own Jira key (e.g. "MLPS-110"), not the epic key ("MLPS-118").
    # They must still be reconciled once linked.
    item = github_item(body="Some unrelated text, no epic mention", state="closed")
    already_prefixed = module.jira.text_to_adf(
        "Automatically synced at: https://github.com/owner/repo/issues/1\n\nanything"
    )
    jira_issue = {
        "fields": {
            "description": already_prefixed,
            "status": {"statusCategory": {"key": "new"}},
        }
    }

    with patch(
        "jira_sync.github_to_jira.jira.search",
        return_value=[{"key": "MLPS-55", "fields": {"description": already_prefixed}}],
    ), patch("jira_sync.github_to_jira.jira.get_issue", return_value=jira_issue), patch(
        "jira_sync.github_to_jira.jira.update_description"
    ), patch("jira_sync.github_to_jira.jira.transition_issue") as transition_issue:
        module.sync_item(item)

    transition_issue.assert_called_once_with("MLPS-55", module.JIRA_DONE_STATUS)


def test_github_to_jira_relinks_when_linked_jira_issue_was_deleted():
    item = github_item(body="Part of MLPS-118")
    already_prefixed = module.jira.text_to_adf(
        "Automatically synced at: https://github.com/owner/repo/issues/1"
    )

    with patch(
        "jira_sync.github_to_jira.jira.search",
        return_value=[{"key": "MLPS-55", "fields": {"description": already_prefixed}}],
    ), patch("jira_sync.github_to_jira.jira.get_issue", return_value=None), patch(
        "jira_sync.github_to_jira.jira.create_issue", return_value=None
    ) as create_issue, patch(
        "jira_sync.github_to_jira.jira.update_description"
    ) as update_description, patch(
        "jira_sync.github_to_jira.jira.transition_issue"
    ) as transition_issue:
        module.sync_item(item)

    # The stale link is dropped once MLPS-55 fails to resolve, so a fresh
    # issue is created instead.
    create_issue.assert_called_once()
    update_description.assert_not_called()
    transition_issue.assert_not_called()


def test_github_to_jira_appends_jira_description_postfix_when_missing():
    item = github_item(number=9, body="Part of MLPS-118 updated")
    tracked_description = "human-written summary\n\n----------------------------------------\nAutomatically synced at: https://github.com/owner/repo/issues/9"
    stale_issue = make_jira_issue(
        description="human-written summary", status_category="new"
    )

    with patch(
        "jira_sync.github_to_jira.jira.search",
        return_value=[
            {
                "key": "MLPS-60",
                "fields": {"description": module.jira.text_to_adf(tracked_description)},
            }
        ],
    ), patch(
        "jira_sync.github_to_jira.jira.get_issue", return_value=stale_issue
    ), patch(
        "jira_sync.github_to_jira.jira.update_description"
    ) as update_description, patch("jira_sync.github_to_jira.jira.transition_issue"):
        module.sync_item(item)

    expected = (
        "human-written summary\n\n"
        "----------------------------------------\n"
        "Automatically synced at: https://github.com/owner/repo/issues/9"
    )
    update_description.assert_called_once_with("MLPS-60", expected)


def test_github_to_jira_does_not_update_jira_description_when_postfix_already_present():
    item = github_item(number=9, body="Part of MLPS-118")
    already_postfixed_text = "human-written summary\n\n----------------------------------------\nAutomatically synced at: https://github.com/owner/repo/issues/9"
    already_postfixed = module.jira.text_to_adf(already_postfixed_text)
    current_issue = make_jira_issue(
        description=already_postfixed_text, status_category="new"
    )

    with patch(
        "jira_sync.github_to_jira.jira.search",
        return_value=[{"key": "MLPS-61", "fields": {"description": already_postfixed}}],
    ), patch(
        "jira_sync.github_to_jira.jira.get_issue", return_value=current_issue
    ), patch(
        "jira_sync.github_to_jira.jira.update_description"
    ) as update_description, patch("jira_sync.github_to_jira.jira.transition_issue"):
        module.sync_item(item)

    update_description.assert_not_called()


def test_creates_jira_issue_for_unlinked_pull_request():
    pull_request = {
        "number": 3,
        "title": "My PR",
        "body": "Fixes MLPS-118",
        "html_url": "https://github.com/owner/repo/pull/3",
    }

    with patch("jira_sync.github_to_jira.jira.search", return_value=[]), patch(
        "jira_sync.github_to_jira.jira.create_issue", return_value="MLPS-102"
    ) as create_issue:
        module.sync_item(pull_request)

    assert create_issue.call_args.kwargs["summary"] == "My PR"
    assert create_issue.call_args.kwargs["description"] == (
        "Fixes MLPS-118\n\n----------------------------------------\nAutomatically synced at: https://github.com/owner/repo/pull/3"
    )


def test_github_to_jira_sync_all_calls_sync_item_for_each_search_result():
    items = [
        {
            "number": 1,
            "title": "A",
            "body": "MLPS-118",
            "html_url": "u1",
            "state": "open",
        },
        {
            "number": 2,
            "title": "B",
            "body": "MLPS-118",
            "html_url": "u2",
            "state": "closed",
            "pull_request": {},
        },
    ]

    with patch(
        "jira_sync.github_to_jira.gh.search_issues", return_value=items
    ) as search_issues, patch(
        "jira_sync.github_to_jira.jira.search", return_value=[]
    ), patch("jira_sync.github_to_jira.sync_item") as sync_item:
        module.sync_all()

    search_issues.assert_called_once()
    assert 'repo:owner/repo "MLPS-118" in:body,title' in search_issues.call_args.args[0]
    assert sync_item.call_args_list == [
        ((items[0],),),
        ((items[1],),),
    ]


def test_github_to_jira_sync_all_skips_the_github_epic_parent():
    epic = {
        "number": 1,
        "body": "Automatically synced at: https://example.atlassian.net/browse/MLPS-118",
    }

    with patch("jira_sync.github_to_jira.gh.search_issues", return_value=[epic]), patch(
        "jira_sync.github_to_jira.jira.search", return_value=[]
    ), patch("jira_sync.github_to_jira.sync_item") as sync_item:
        module.sync_all()

    sync_item.assert_not_called()


def test_github_to_jira_sync_all_also_discovers_items_via_jira_description_link():
    gh_item = {"number": 5, "title": "C", "body": "unrelated now", "state": "open"}
    jira_issues = [
        {
            "fields": {
                "description": module.jira.text_to_adf(
                    "Automatically synced at: https://github.com/owner/repo/issues/5"
                )
            }
        }
    ]

    with patch("jira_sync.github_to_jira.gh.search_issues", return_value=[]), patch(
        "jira_sync.github_to_jira.jira.search", return_value=jira_issues
    ) as jira_search, patch(
        "jira_sync.github_to_jira.gh.get_issue", return_value=gh_item
    ), patch("jira_sync.github_to_jira.sync_item") as sync_item:
        module.sync_all()

    jira_search.assert_called_once_with(module.JIRA_JQL, fields=["description"])
    sync_item.assert_called_once_with(gh_item)


def test_github_to_jira_sync_all_does_not_duplicate_items_found_both_ways():
    gh_item = {
        "number": 5,
        "title": "C",
        "body": "MLPS-118",
        "html_url": "u5",
        "state": "open",
    }
    jira_issues = [
        {
            "fields": {
                "description": module.jira.text_to_adf(
                    "Automatically synced at: https://github.com/owner/repo/issues/5"
                )
            }
        }
    ]

    with patch(
        "jira_sync.github_to_jira.gh.search_issues", return_value=[gh_item]
    ), patch("jira_sync.github_to_jira.jira.search", return_value=jira_issues), patch(
        "jira_sync.github_to_jira.gh.get_issue"
    ) as get_issue, patch("jira_sync.github_to_jira.sync_item") as sync_item:
        module.sync_all()

    get_issue.assert_not_called()
    sync_item.assert_called_once_with(gh_item)
