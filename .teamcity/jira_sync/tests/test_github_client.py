from unittest.mock import MagicMock, patch

from jira_sync import github_client as gh


def _mock_response(ok=True, status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data if json_data is not None else {}
    return resp


@patch("jira_sync.github_client.requests.request")
def test_get_issue_returns_json_on_success(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(
        json_data={"number": 42, "state": "open"}
    )

    # Act
    result = gh.get_issue(42)

    # Assert
    assert result == {"number": 42, "state": "open"}
    mock_request.assert_called_once_with(
        "GET",
        "https://api.github.com/repos/owner/repo/issues/42",
        headers=gh._headers(),
    )


@patch("jira_sync.github_client.requests.request")
def test_get_issue_returns_none_on_failure(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(
        ok=False, status_code=404, text="not found"
    )

    # Act
    result = gh.get_issue(999)

    # Assert
    assert result is None


@patch("jira_sync.github_client.requests.request")
def test_create_issue_returns_number_on_success(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(json_data={"number": 13})

    # Act
    number = gh.create_issue("title", "body")

    # Assert
    assert number == 13
    assert mock_request.call_args.kwargs["json"] == {"title": "title", "body": "body"}


@patch("jira_sync.github_client.requests.request")
def test_create_issue_returns_none_on_failure(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(ok=False)

    # Act
    result = gh.create_issue("title", "body")

    # Assert
    assert result is None


@patch("jira_sync.github_client.requests.request")
def test_list_comments_returns_list(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(json_data=[{"body": "Jira: PROJ-1"}])

    # Act
    comments = gh.list_comments(1)

    # Assert
    assert comments == [{"body": "Jira: PROJ-1"}]


@patch("jira_sync.github_client.requests.request")
def test_list_comments_returns_empty_on_failure(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(ok=False)

    # Act
    comments = gh.list_comments(1)

    # Assert
    assert comments == []


@patch("jira_sync.github_client.requests.request")
def test_add_comment_posts_body(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(ok=True)

    # Act
    result = gh.add_comment(1, "hello")

    # Assert
    assert result is True
    assert mock_request.call_args.kwargs["json"] == {"body": "hello"}


@patch("jira_sync.github_client.requests.request")
def test_set_state_patches_state(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(ok=True)

    # Act
    result = gh.set_state(1, "closed")

    # Assert
    assert result is True
    assert mock_request.call_args.kwargs["json"] == {"state": "closed"}
    assert mock_request.call_args.args[0] == "PATCH"


@patch("jira_sync.github_client.requests.request")
def test_update_issue_patches_given_fields(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(ok=True)

    # Act
    result = gh.update_issue(1, body="new body")

    # Assert
    assert result is True
    assert mock_request.call_args.kwargs["json"] == {"body": "new body"}
    assert mock_request.call_args.args[0] == "PATCH"


@patch("jira_sync.github_client.requests.request")
def test_list_sub_issue_ids_returns_ids(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(json_data=[{"id": 11}, {"id": 12}])

    # Act
    issue_ids = gh.list_sub_issue_ids(1)

    # Assert
    assert issue_ids == {11, 12}


@patch("jira_sync.github_client.requests.request")
def test_list_sub_issues_paginates(mock_request):
    # Arrange
    first_page = [{"id": number} for number in range(100)]
    second_page = [{"id": 100, "node_id": "I_kwDO-test"}]
    mock_request.side_effect = [
        _mock_response(json_data=first_page),
        _mock_response(json_data=second_page),
    ]

    # Act
    issues = gh.list_sub_issues(1)

    # Assert
    assert issues[-1] == second_page[0]
    assert mock_request.call_count == 2
    assert mock_request.call_args_list[1].kwargs["params"] == {
        "per_page": 100,
        "page": 2,
    }


@patch("jira_sync.github_client.requests.request")
def test_add_sub_issue_posts_child_id(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(ok=True)

    # Act
    result = gh.add_sub_issue(1, 12)

    # Assert
    assert result is True
    assert mock_request.call_args.kwargs["json"] == {"sub_issue_id": 12}


@patch("jira_sync.github_client.requests.request")
def test_delete_issue_uses_graphql_node_id(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(
        json_data={"data": {"deleteIssue": {"clientMutationId": None}}}
    )

    # Act
    result = gh.delete_issue("I_kwDO-test")

    # Assert
    assert result is True
    mock_request.assert_called_once()
    assert mock_request.call_args.args[:2] == (
        "POST",
        "https://api.github.com/graphql",
    )
    assert mock_request.call_args.kwargs["json"]["variables"] == {
        "issueId": "I_kwDO-test"
    }


@patch("jira_sync.github_client.requests.request")
def test_delete_issue_returns_false_on_graphql_error(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(
        json_data={"errors": [{"message": "Insufficient permissions"}]}
    )

    # Act
    result = gh.delete_issue("I_kwDO-test")

    # Assert
    assert result is False


@patch("jira_sync.github_client.requests.request")
def test_search_issues_returns_items(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(json_data={"items": [{"number": 1}]})

    # Act
    items = gh.search_issues("repo:owner/repo MLPS-118")

    # Assert
    assert items == [{"number": 1}]
    assert mock_request.call_args.kwargs["params"]["q"] == "repo:owner/repo MLPS-118"


@patch("jira_sync.github_client.requests.request")
def test_search_issues_paginates(mock_request):
    # Arrange
    page_1 = [{"number": n} for n in range(100)]
    page_2 = [{"number": 100}]
    mock_request.side_effect = [
        _mock_response(json_data={"items": page_1}),
        _mock_response(json_data={"items": page_2}),
    ]

    # Act
    items = gh.search_issues("repo:owner/repo MLPS-118")

    # Assert
    assert len(items) == 101
    assert mock_request.call_count == 2


@patch("jira_sync.github_client.requests.request")
def test_search_issues_returns_empty_on_failure(mock_request):
    # Arrange
    mock_request.return_value = _mock_response(ok=False)

    # Act
    items = gh.search_issues("repo:owner/repo MLPS-118")

    # Assert
    assert items == []


def test_headers_include_bearer_token(monkeypatch):
    # Arrange
    monkeypatch.setenv("GITHUB_TOKEN", "abc123")

    # Act
    headers = gh._headers()

    # Assert
    assert headers["Authorization"] == "Bearer abc123"
