import inspect
from unittest.mock import MagicMock, patch

from jira_sync import jira_client


def _mock_response(ok=True, status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data or {}
    return resp


def test_url_joins_base_and_path():
    assert (
        jira_client._url("issue/ABC-1")
        == "https://example.atlassian.net/rest/api/3/issue/ABC-1"
    )
    assert (
        jira_client._url("/issue/ABC-1")
        == "https://example.atlassian.net/rest/api/3/issue/ABC-1"
    )


def test_auth_reads_env(monkeypatch):
    monkeypatch.setenv("JIRA_EMAIL", "me@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "secret")
    assert jira_client._auth() == ("me@example.com", "secret")


def test_text_to_adf_wraps_plain_text():
    adf = jira_client.text_to_adf("hello")
    assert adf["content"][0]["content"][0]["text"] == "hello"


def test_adf_to_text_extracts_paragraph_text():
    adf = jira_client.text_to_adf("test description")
    assert jira_client.adf_to_text(adf) == "test description"


def test_adf_to_text_handles_plain_string():
    assert jira_client.adf_to_text("already plain") == "already plain"


def test_adf_to_text_handles_none_and_empty():
    assert jira_client.adf_to_text(None) == ""
    assert jira_client.adf_to_text({}) == ""


@patch("jira_sync.jira_client.requests.request")
def test_create_issue_with_parent_success(mock_request):
    mock_request.return_value = _mock_response(json_data={"key": "PROJ-5"})

    key = jira_client.create_issue(
        project_key="PROJ",
        issue_type="Task",
        summary="Summary",
        description="Desc",
        epic_key="PROJ-98",
    )

    assert key == "PROJ-5"
    sent_fields = mock_request.call_args.kwargs["json"]["fields"]
    assert sent_fields["parent"] == {"key": "PROJ-98"}
    assert mock_request.call_count == 1


@patch("jira_sync.jira_client.requests.request")
def test_create_issue_falls_back_without_parent(mock_request):
    mock_request.side_effect = [
        _mock_response(ok=False, status_code=400, text="invalid parent"),
        _mock_response(ok=True, json_data={"key": "PROJ-6"}),
    ]

    key = jira_client.create_issue(
        project_key="PROJ",
        issue_type="Task",
        summary="Summary",
        description="Desc",
        epic_key="PROJ-98",
    )

    assert key == "PROJ-6"
    assert mock_request.call_count == 2
    retried_fields = mock_request.call_args.kwargs["json"]["fields"]
    assert "parent" not in retried_fields


def test_create_issue_signature_has_only_supported_epic_argument():
    params = inspect.signature(jira_client.create_issue).parameters
    assert "epic_link_field" not in params


@patch("jira_sync.jira_client.requests.request")
def test_search_paginates_until_last_page(mock_request):
    mock_request.side_effect = [
        _mock_response(
            json_data={
                "issues": [{"key": "A-1"}],
                "isLast": False,
                "nextPageToken": "tok",
            }
        ),
        _mock_response(json_data={"issues": [{"key": "A-2"}], "isLast": True}),
    ]

    issues = jira_client.search("project = A", fields=["summary"], max_results=1)

    assert [i["key"] for i in issues] == ["A-1", "A-2"]
    assert mock_request.call_count == 2
    assert mock_request.call_args_list[1].kwargs["params"]["nextPageToken"] == "tok"
    assert "search/jql" in mock_request.call_args_list[0].args[1]


@patch("jira_sync.jira_client.requests.request")
def test_transition_issue_finds_matching_transition(mock_request):
    mock_request.side_effect = [
        _mock_response(
            json_data={
                "transitions": [
                    {"id": "1", "to": {"name": "To Do"}},
                    {"id": "2", "to": {"name": "Done"}},
                ]
            }
        ),
        _mock_response(ok=True),
    ]

    assert jira_client.transition_issue("PROJ-1", "done") is True
    transition_call = mock_request.call_args
    assert transition_call.kwargs["json"] == {"transition": {"id": "2"}}


@patch("jira_sync.jira_client.requests.request")
def test_transition_issue_no_match_returns_false(mock_request):
    mock_request.return_value = _mock_response(json_data={"transitions": []})

    assert jira_client.transition_issue("PROJ-1", "Done") is False
    assert mock_request.call_count == 1
