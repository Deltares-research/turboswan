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
    mock_request.return_value = _mock_response(
        json_data={"number": 42, "state": "open"}
    )

    result = gh.get_issue(42)

    assert result == {"number": 42, "state": "open"}
    mock_request.assert_called_once_with(
        "GET",
        "https://api.github.com/repos/owner/repo/issues/42",
        headers=gh._headers(),
    )


@patch("jira_sync.github_client.requests.request")
def test_get_issue_returns_none_on_failure(mock_request):
    mock_request.return_value = _mock_response(
        ok=False, status_code=404, text="not found"
    )

    assert gh.get_issue(999) is None


@patch("jira_sync.github_client.requests.request")
def test_create_issue_returns_number_on_success(mock_request):
    mock_request.return_value = _mock_response(json_data={"number": 13})

    number = gh.create_issue("title", "body")

    assert number == 13
    assert mock_request.call_args.kwargs["json"] == {"title": "title", "body": "body"}


@patch("jira_sync.github_client.requests.request")
def test_create_issue_returns_none_on_failure(mock_request):
    mock_request.return_value = _mock_response(ok=False)

    assert gh.create_issue("title", "body") is None


@patch("jira_sync.github_client.requests.request")
def test_list_comments_returns_list(mock_request):
    mock_request.return_value = _mock_response(json_data=[{"body": "Jira: PROJ-1"}])

    comments = gh.list_comments(1)

    assert comments == [{"body": "Jira: PROJ-1"}]


@patch("jira_sync.github_client.requests.request")
def test_list_comments_returns_empty_on_failure(mock_request):
    mock_request.return_value = _mock_response(ok=False)

    assert gh.list_comments(1) == []


@patch("jira_sync.github_client.requests.request")
def test_add_comment_posts_body(mock_request):
    mock_request.return_value = _mock_response(ok=True)

    assert gh.add_comment(1, "hello") is True
    assert mock_request.call_args.kwargs["json"] == {"body": "hello"}


@patch("jira_sync.github_client.requests.request")
def test_set_state_patches_state(mock_request):
    mock_request.return_value = _mock_response(ok=True)

    assert gh.set_state(1, "closed") is True
    assert mock_request.call_args.kwargs["json"] == {"state": "closed"}
    assert mock_request.call_args.args[0] == "PATCH"


@patch("jira_sync.github_client.requests.request")
def test_update_issue_patches_given_fields(mock_request):
    mock_request.return_value = _mock_response(ok=True)

    assert gh.update_issue(1, body="new body") is True
    assert mock_request.call_args.kwargs["json"] == {"body": "new body"}
    assert mock_request.call_args.args[0] == "PATCH"


@patch("jira_sync.github_client.requests.request")
def test_list_sub_issue_ids_returns_ids(mock_request):
    mock_request.return_value = _mock_response(json_data=[{"id": 11}, {"id": 12}])

    assert gh.list_sub_issue_ids(1) == {11, 12}


@patch("jira_sync.github_client.requests.request")
def test_add_sub_issue_posts_child_id(mock_request):
    mock_request.return_value = _mock_response(ok=True)

    assert gh.add_sub_issue(1, 12) is True
    assert mock_request.call_args.kwargs["json"] == {"sub_issue_id": 12}


@patch("jira_sync.github_client.requests.request")
def test_search_issues_returns_items(mock_request):
    mock_request.return_value = _mock_response(json_data={"items": [{"number": 1}]})

    items = gh.search_issues("repo:owner/repo MLPS-118")

    assert items == [{"number": 1}]
    assert mock_request.call_args.kwargs["params"]["q"] == "repo:owner/repo MLPS-118"


@patch("jira_sync.github_client.requests.request")
def test_search_issues_paginates(mock_request):
    page_1 = [{"number": n} for n in range(100)]
    page_2 = [{"number": 100}]
    mock_request.side_effect = [
        _mock_response(json_data={"items": page_1}),
        _mock_response(json_data={"items": page_2}),
    ]

    items = gh.search_issues("repo:owner/repo MLPS-118")

    assert len(items) == 101
    assert mock_request.call_count == 2


@patch("jira_sync.github_client.requests.request")
def test_search_issues_returns_empty_on_failure(mock_request):
    mock_request.return_value = _mock_response(ok=False)

    assert gh.search_issues("repo:owner/repo MLPS-118") == []


def test_headers_include_bearer_token(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "abc123")
    headers = gh._headers()
    assert headers["Authorization"] == "Bearer abc123"
