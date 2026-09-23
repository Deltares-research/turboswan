from jira_sync.linking import (
    append_sync_footer,
    build_jira_description,
    ensure_jira_description_postfix,
    parse_gh_number,
    remove_sync_footer,
    SYNC_SEPARATOR,
    sync_footer,
)


def test_sync_footer_format():
    assert (
        sync_footer("https://github.com/owner/repo/issues/1")
        == "Automatically synced at: https://github.com/owner/repo/issues/1"
    )


def test_append_sync_footer_adds_separator_after_description():
    assert append_sync_footer("description", "https://example.test/issue/1") == (
        f"description\n\n{SYNC_SEPARATOR}\n" "Automatically synced at: https://example.test/issue/1"
    )


def test_build_jira_description_uses_tracked_link_as_postfix():
    description = build_jira_description("https://github.com/owner/repo/issues/1", "")
    assert description == "Automatically synced at: https://github.com/owner/repo/issues/1"


def test_build_jira_description_appends_tracking_link_after_body():
    description = build_jira_description(
        "https://github.com/owner/repo/issues/1", "original issue text"
    )
    assert description == (
        "original issue text\n\n"
        f"{SYNC_SEPARATOR}\nAutomatically synced at: https://github.com/owner/repo/issues/1"
    )


def test_ensure_jira_description_postfix_appends_when_missing():
    result = ensure_jira_description_postfix(
        "human-written summary", "https://github.com/owner/repo/issues/1"
    )
    assert result == (
        "human-written summary\n\n"
        f"{SYNC_SEPARATOR}\nAutomatically synced at: https://github.com/owner/repo/issues/1"
    )


def test_ensure_jira_description_postfix_handles_empty_description():
    result = ensure_jira_description_postfix(
        "", "https://github.com/owner/repo/issues/1"
    )
    assert result == "Automatically synced at: https://github.com/owner/repo/issues/1"


def test_ensure_jira_description_postfix_replaces_existing_footer():
    current = "Automatically synced at: https://github.com/owner/repo/issues/99\n\nsummary"
    result = ensure_jira_description_postfix(
        current, "https://github.com/owner/repo/issues/1"
    )
    assert result == (
        f"summary\n\n{SYNC_SEPARATOR}\n"
        "Automatically synced at: https://github.com/owner/repo/issues/1"
    )


def test_remove_sync_footer_preserves_content():
    assert (
        remove_sync_footer(
            "Automatically synced at: https://github.com/owner/repo/issues/1\n\nsummary"
        )
        == "summary"
    )


def test_parse_gh_number_extracts_issue_number():
    assert parse_gh_number("Automatically synced at: https://github.com/owner/repo/issues/42") == 42


def test_parse_gh_number_extracts_pull_number():
    assert parse_gh_number("Automatically synced at: https://github.com/owner/repo/pull/7") == 7


def test_parse_gh_number_returns_none_without_url():
    assert parse_gh_number("plain summary") is None


def test_parse_gh_number_handles_none():
    assert parse_gh_number(None) is None
