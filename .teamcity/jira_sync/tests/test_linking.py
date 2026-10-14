from jira_sync.linking import (
    SYNC_SEPARATOR,
    append_sync_footer,
    build_jira_description,
    ensure_jira_description_postfix,
    parse_gh_number,
    remove_sync_footer,
    sync_footer,
)


def test_sync_footer_format():
    # Arrange
    issue_url = "https://github.com/owner/repo/issues/1"

    # Act
    footer = sync_footer(issue_url)

    # Assert
    assert footer == "Automatically synced at: https://github.com/owner/repo/issues/1"


def test_append_sync_footer_adds_separator_after_description():
    # Arrange
    description = "description"
    issue_url = "https://example.test/issue/1"

    # Act
    result = append_sync_footer(description, issue_url)

    # Assert
    assert result == (
        f"description\n\n{SYNC_SEPARATOR}\n"
        "Automatically synced at: https://example.test/issue/1"
    )


def test_build_jira_description_uses_tracked_link_as_postfix():
    # Arrange
    issue_url = "https://github.com/owner/repo/issues/1"

    # Act
    description = build_jira_description(issue_url, "")

    # Assert
    assert (
        description == "Automatically synced at: https://github.com/owner/repo/issues/1"
    )


def test_build_jira_description_appends_tracking_link_after_body():
    # Arrange
    issue_url = "https://github.com/owner/repo/issues/1"
    original_description = "original issue text"

    # Act
    description = build_jira_description(issue_url, original_description)

    # Assert
    assert description == (
        "original issue text\n\n"
        f"{SYNC_SEPARATOR}\nAutomatically synced at: https://github.com/owner/repo/issues/1"
    )


def test_ensure_jira_description_postfix_appends_when_missing():
    # Arrange
    description = "human-written summary"
    issue_url = "https://github.com/owner/repo/issues/1"

    # Act
    result = ensure_jira_description_postfix(description, issue_url)

    # Assert
    assert result == (
        "human-written summary\n\n"
        f"{SYNC_SEPARATOR}\nAutomatically synced at: https://github.com/owner/repo/issues/1"
    )


def test_ensure_jira_description_postfix_handles_empty_description():
    # Arrange
    description = ""
    issue_url = "https://github.com/owner/repo/issues/1"

    # Act
    result = ensure_jira_description_postfix(description, issue_url)

    # Assert
    assert result == "Automatically synced at: https://github.com/owner/repo/issues/1"


def test_ensure_jira_description_postfix_replaces_existing_footer():
    # Arrange
    current = (
        "Automatically synced at: https://github.com/owner/repo/issues/99\n\nsummary"
    )
    issue_url = "https://github.com/owner/repo/issues/1"

    # Act
    result = ensure_jira_description_postfix(current, issue_url)

    # Assert
    assert result == (
        f"summary\n\n{SYNC_SEPARATOR}\n"
        "Automatically synced at: https://github.com/owner/repo/issues/1"
    )


def test_remove_sync_footer_preserves_content():
    # Arrange
    description = (
        "Automatically synced at: https://github.com/owner/repo/issues/1\n\nsummary"
    )

    # Act
    result = remove_sync_footer(description)

    # Assert
    assert result == "summary"


def test_parse_gh_number_extracts_issue_number():
    # Arrange
    description = "Automatically synced at: https://github.com/owner/repo/issues/42"

    # Act
    number = parse_gh_number(description)

    # Assert
    assert number == 42


def test_parse_gh_number_extracts_pull_number():
    # Arrange
    description = "Automatically synced at: https://github.com/owner/repo/pull/7"

    # Act
    number = parse_gh_number(description)

    # Assert
    assert number == 7


def test_parse_gh_number_returns_none_without_url():
    # Arrange
    description = "plain summary"

    # Act
    number = parse_gh_number(description)

    # Assert
    assert number is None


def test_parse_gh_number_handles_none():
    # Arrange
    description = None

    # Act
    number = parse_gh_number(description)

    # Assert
    assert number is None
