from unittest.mock import MagicMock, patch

import pytest
import requests
import jira_sync.jira_client as jira_client


def _mock_response(ok=True, status_code=200, json_data=None, text=""):
    resp = MagicMock()
    resp.ok = ok
    resp.status_code = status_code
    resp.text = text
    resp.json.return_value = json_data or {}
    return resp


def test_url_joins_base_and_path():
    # Arrange
    paths = ["issue/ABC-1", "/issue/ABC-1"]

    # Act
    urls = [jira_client._url(path) for path in paths]

    # Assert
    expected_url = "https://example.atlassian.net/rest/api/3/issue/ABC-1"
    assert urls == [expected_url, expected_url]


def test_auth_reads_env(monkeypatch):
    # Arrange
    monkeypatch.setenv("JIRA_EMAIL", "me@example.com")
    monkeypatch.setenv("JIRA_API_TOKEN", "secret")

    # Act
    auth = jira_client._auth()

    # Assert
    assert auth == ("me@example.com", "secret")


def test_text_to_adf_wraps_plain_text():
    # Arrange
    text = "hello"

    # Act
    adf = jira_client.text_to_adf(text)

    # Assert
    assert adf["content"][0]["content"][0]["text"] == "hello"


def test_adf_to_text_extracts_paragraph_text():
    # Arrange
    adf = jira_client.text_to_adf("test description")

    # Act
    text = jira_client.adf_to_text(adf)

    # Assert
    assert text == "test description"


def test_adf_to_text_handles_plain_string():
    # Arrange
    plain_text = "already plain"

    # Act
    result = jira_client.adf_to_text(plain_text)

    # Assert
    assert result == plain_text


def test_adf_to_text_handles_none_and_empty():
    # Arrange
    values = [None, {}]

    # Act
    results = [jira_client.adf_to_text(value) for value in values]

    # Assert
    assert results == ["", ""]


@patch("jira_sync.jira_client.requests.request")
def test_search_paginates_until_last_page(mock_request):
    # Arrange
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

    # Act
    issues = jira_client.search("project = A", fields=["summary"], max_results=1)

    # Assert
    assert [i["key"] for i in issues] == ["A-1", "A-2"]
    assert mock_request.call_count == 2
    assert mock_request.call_args_list[1].kwargs["params"]["nextPageToken"] == "tok"
    assert "search/jql" in mock_request.call_args_list[0].args[1]


@patch("jira_sync.jira_client.requests.request")
@pytest.mark.parametrize("after_first_page", [False, True])
def test_search_failure_does_not_return_empty_or_partial_results(
    mock_request, after_first_page
):
    # Arrange
    failure = _mock_response(ok=False, status_code=403, text="Forbidden")
    failure.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
    responses = []
    if after_first_page:
        responses.append(
            _mock_response(
                json_data={
                    "issues": [{"key": "MLPS-1"}],
                    "isLast": False,
                    "nextPageToken": "tok",
                }
            )
        )
    mock_request.side_effect = responses + [failure]

    # Act and assert
    with pytest.raises(requests.HTTPError, match="403 Forbidden"):
        jira_client.search('parent = "MLPS-118"', fields=["summary"])

    # Assert
    assert mock_request.call_count == len(responses) + 1
