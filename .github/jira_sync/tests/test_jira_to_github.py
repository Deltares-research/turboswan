from unittest.mock import patch

import pytest

from jira_sync import jira_to_github as module


def make_issue(key, description, status_category, summary="Summary"):
    return {
        "key": key,
        "fields": {
            "status": {"statusCategory": {"key": status_category}},
            "summary": summary,
            "description": description,
        },
    }


def tracked(number, extra=""):
    text = f"Automatically synced at: https://github.com/owner/repo/issues/{number}"
    if extra:
        text = f"{extra}\n\n{text}"
    return text


def make_jira_comment(comment_id, text, author="Jira user"):
    return {"id": comment_id, "author": {"displayName": author}, "body": text}


@pytest.fixture(autouse=True)
def skip_epic_parent_sync(request):
    if "epic_parent" in request.node.name:
        yield
        return
    with patch("jira_sync.jira_to_github.ensure_epic_issue", return_value=None):
        yield


def test_jira_to_github_sync_comments_mirrors_new_jira_comment():
    with patch("jira_sync.jira_to_github.gh.list_comments", return_value=[]), patch(
        "jira_sync.jira_to_github.jira.list_comments",
        return_value=[make_jira_comment("10", "hello there", author="Alice")],
    ), patch("jira_sync.jira_to_github.gh.add_comment") as add_comment:
        module.sync_comments("MLPS-1", 5)

    add_comment.assert_called_once()
    posted_body = add_comment.call_args.args[1]
    assert posted_body.startswith("_Synced from Jira comment by Alice:_")
    assert "hello there" in posted_body
    assert "<!-- jira-comment-id:10 -->" in posted_body


def test_jira_to_github_sync_comments_skips_already_mirrored_comment():
    gh_comments = [
        {
            "body": "_Synced from Jira comment by Alice:_\n\nhi\n\n<!-- jira-comment-id:10 -->"
        }
    ]

    with patch(
        "jira_sync.jira_to_github.gh.list_comments", return_value=gh_comments
    ), patch(
        "jira_sync.jira_to_github.jira.list_comments",
        return_value=[make_jira_comment("10", "hi", author="Alice")],
    ), patch("jira_sync.jira_to_github.gh.add_comment") as add_comment:
        module.sync_comments("MLPS-1", 5)

    add_comment.assert_not_called()


def test_jira_to_github_sync_comments_skips_comments_that_originated_on_github():
    with patch("jira_sync.jira_to_github.gh.list_comments", return_value=[]), patch(
        "jira_sync.jira_to_github.jira.list_comments",
        return_value=[
            make_jira_comment("11", "Synced from GitHub comment by bob:\n\noriginal")
        ],
    ), patch("jira_sync.jira_to_github.gh.add_comment") as add_comment:
        module.sync_comments("MLPS-1", 5)

    add_comment.assert_not_called()


def test_jira_to_github_closes_open_gh_issue_when_jira_done():
    issues = [make_issue("MLPS-1", tracked(5), "done")]

    with patch("jira_sync.jira_to_github.jira.search", return_value=issues), patch(
        "jira_sync.jira_to_github.gh.get_issue",
        return_value={"state": "open", "body": ""},
    ), patch("jira_sync.jira_to_github.gh.update_issue"), patch(
        "jira_sync.jira_to_github.sync_comments"
    ), patch("jira_sync.jira_to_github.gh.add_comment") as add_comment, patch(
        "jira_sync.jira_to_github.gh.set_state"
    ) as set_state:
        module.sync_all()

    set_state.assert_called_once_with(5, "closed")
    add_comment.assert_called_once()


def test_jira_to_github_reopens_closed_gh_issue_when_jira_not_done():
    issues = [make_issue("MLPS-2", tracked(6), "indeterminate")]

    with patch("jira_sync.jira_to_github.jira.search", return_value=issues), patch(
        "jira_sync.jira_to_github.gh.get_issue",
        return_value={"state": "closed", "body": ""},
    ), patch("jira_sync.jira_to_github.gh.update_issue"), patch(
        "jira_sync.jira_to_github.sync_comments"
    ), patch("jira_sync.jira_to_github.gh.add_comment") as add_comment, patch(
        "jira_sync.jira_to_github.gh.set_state"
    ) as set_state:
        module.sync_all()

    set_state.assert_called_once_with(6, "open")
    add_comment.assert_called_once()


def test_jira_to_github_no_change_when_states_already_match():
    issues = [make_issue("MLPS-3", tracked(8), "done")]

    with patch("jira_sync.jira_to_github.jira.search", return_value=issues), patch(
        "jira_sync.jira_to_github.gh.get_issue",
        return_value={"state": "closed", "body": ""},
    ), patch("jira_sync.jira_to_github.gh.update_issue"), patch(
        "jira_sync.jira_to_github.sync_comments"
    ), patch("jira_sync.jira_to_github.gh.set_state") as set_state:
        module.sync_all()

    set_state.assert_not_called()


def test_jira_to_github_sync_all_syncs_comments_for_each_linked_issue():
    issues = [make_issue("MLPS-9", tracked(14), "done")]

    with patch("jira_sync.jira_to_github.jira.search", return_value=issues), patch(
        "jira_sync.jira_to_github.gh.get_issue",
        return_value={"state": "closed", "body": ""},
    ), patch("jira_sync.jira_to_github.gh.update_issue"), patch(
        "jira_sync.jira_to_github.sync_comments"
    ) as sync_comments:
        module.sync_all()

    sync_comments.assert_called_once_with("MLPS-9", 14)


def test_jira_to_github_creates_issue_when_description_has_no_gh_link():
    issues = [make_issue("MLPS-4", "Unrelated description", "done")]

    with patch("jira_sync.jira_to_github.jira.search", return_value=issues), patch(
        "jira_sync.jira_to_github.gh.create_issue", return_value=None
    ) as create_issue, patch("jira_sync.jira_to_github.gh.get_issue") as get_issue:
        module.sync_all()

    create_issue.assert_called_once()
    get_issue.assert_not_called()


def test_jira_to_github_creates_and_links_gh_issue_when_unlinked():
    issues = [make_issue("MLPS-6", "test description", "new", summary="Summary")]

    with patch("jira_sync.jira_to_github.jira.search", return_value=issues), patch(
        "jira_sync.jira_to_github.gh.create_issue", return_value=11
    ) as create_issue, patch(
        "jira_sync.jira_to_github.jira.update_description"
    ) as update_description, patch(
        "jira_sync.jira_to_github.gh.get_issue",
        return_value={"state": "open", "body": None},
    ), patch("jira_sync.jira_to_github.gh.update_issue") as update_issue, patch(
        "jira_sync.jira_to_github.gh.add_comment"
    ) as add_comment, patch("jira_sync.jira_to_github.sync_comments"):
        module.sync_all()

    create_issue.assert_called_once()
    assert create_issue.call_args.kwargs["title"] == "Summary"
    expected_url = f"{module.os.environ['JIRA_BASE_URL'].rstrip('/')}/browse/MLPS-6"
    expected_body = f"test description\n\n----------------------------------------\nAutomatically synced at: {expected_url}"
    assert create_issue.call_args.kwargs["body"] == expected_body
    # The GH link is recorded by appending "Automatically synced at: <url>" to
    # the Jira description; no label and no GitHub-side comment are used.
    update_description.assert_called_once_with(
        "MLPS-6",
        "test description\n\n----------------------------------------\nAutomatically synced at: https://github.com/owner/repo/issues/11",
    )
    add_comment.assert_not_called()
    # Freshly created GitHub issue's body (None here) doesn't match the fetched
    # snapshot, so the sync still pushes the body once to reconcile the mock.
    update_issue.assert_called_once_with(11, body=expected_body)


def test_jira_to_github_recreates_and_relinks_missing_gh_issue():
    issues = [make_issue("MLPS-6", tracked(9, "test description"), "new")]

    with patch("jira_sync.jira_to_github.jira.search", return_value=issues), patch(
        "jira_sync.jira_to_github.gh.get_issue",
        side_effect=[None, {"state": "open", "body": None}],
    ), patch(
        "jira_sync.jira_to_github.gh.create_issue", return_value=11
    ) as create_issue, patch(
        "jira_sync.jira_to_github.jira.update_description"
    ) as update_description, patch("jira_sync.jira_to_github.gh.update_issue"), patch(
        "jira_sync.jira_to_github.sync_comments"
    ):
        module.sync_all()

    create_issue.assert_called_once()
    assert create_issue.call_args.kwargs["title"] == "Summary"
    assert "issues/9" not in create_issue.call_args.kwargs["body"]
    update_description.assert_called_once_with(
        "MLPS-6",
        "test description\n\n----------------------------------------\nAutomatically synced at: https://github.com/owner/repo/issues/11",
    )


def test_jira_to_github_creates_epic_parent_and_links_child_issue():
    issues = [make_issue("MLPS-6", tracked(5), "new")]
    epic = make_issue("MLPS-118", "epic description", "new", summary="Epic summary")
    parent = {"number": 99, "id": 990, "title": "Epic summary", "body": ""}
    child = {"number": 5, "id": 50, "state": "open", "body": ""}

    with patch("jira_sync.jira_to_github.jira.search", return_value=issues), patch(
        "jira_sync.jira_to_github.jira.get_issue", return_value=epic
    ), patch("jira_sync.jira_to_github.gh.search_issues", return_value=[]), patch(
        "jira_sync.jira_to_github.gh.create_issue", return_value=99
    ) as create_issue, patch(
        "jira_sync.jira_to_github.jira.update_description"
    ) as update_description, patch(
        "jira_sync.jira_to_github.gh.get_issue", side_effect=[parent, child]
    ), patch(
        "jira_sync.jira_to_github.gh.list_sub_issue_ids", return_value=set()
    ), patch("jira_sync.jira_to_github.gh.add_sub_issue") as add_sub_issue, patch(
        "jira_sync.jira_to_github.gh.update_issue"
    ), patch("jira_sync.jira_to_github.sync_comments"):
        module.sync_all()

    assert create_issue.call_args.args[0] == "Epic summary"
    assert create_issue.call_args.args[1] == (
        "epic description\n\n----------------------------------------\n"
        "Automatically synced at: https://example.atlassian.net/browse/MLPS-118"
    )
    update_description.assert_called_once_with(
        "MLPS-118",
        "epic description\n\n----------------------------------------\n"
        "Automatically synced at: https://example.atlassian.net/browse/MLPS-118",
    )
    add_sub_issue.assert_called_once_with(99, 50)


def test_jira_to_github_updates_gh_issue_body_when_out_of_sync():
    issues = [make_issue("MLPS-7", tracked(12, "updated description"), "new")]

    with patch("jira_sync.jira_to_github.jira.search", return_value=issues), patch(
        "jira_sync.jira_to_github.gh.get_issue",
        return_value={"state": "open", "body": "stale body"},
    ), patch("jira_sync.jira_to_github.gh.update_issue") as update_issue, patch(
        "jira_sync.jira_to_github.sync_comments"
    ):
        module.sync_all()

    expected_url = f"{module.os.environ['JIRA_BASE_URL'].rstrip('/')}/browse/MLPS-7"
    expected_body = f"updated description\n\n----------------------------------------\nAutomatically synced at: {expected_url}"
    update_issue.assert_called_once_with(12, body=expected_body)


def test_jira_to_github_does_not_update_gh_issue_body_when_already_in_sync():
    expected_url_base = "https://example.atlassian.net/browse/MLPS-8"
    issues = [make_issue("MLPS-8", tracked(13, "same"), "new")]

    with patch("jira_sync.jira_to_github.jira.search", return_value=issues), patch(
        "jira_sync.jira_to_github.gh.get_issue",
        return_value={
            "state": "open",
            "body": f"same\n\n----------------------------------------\nAutomatically synced at: {expected_url_base}",
        },
    ), patch("jira_sync.jira_to_github.gh.update_issue") as update_issue, patch(
        "jira_sync.jira_to_github.sync_comments"
    ):
        module.sync_all()

    update_issue.assert_not_called()


def test_skips_when_gh_issue_missing():
    issues = [make_issue("MLPS-5", tracked(9), "done")]

    with patch("jira_sync.jira_to_github.jira.search", return_value=issues), patch(
        "jira_sync.jira_to_github.gh.get_issue", return_value=None
    ), patch("jira_sync.jira_to_github.gh.set_state") as set_state:
        module.sync_all()

    set_state.assert_not_called()
