"""Minimal Jira Cloud REST API v3 client used by the epic-sync workflows."""

from __future__ import annotations

import os
import re
import sys
import time
from typing import Any

import requests

URL_RE = re.compile(r"https?://\S+")
GET_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1


def _base_url() -> str:
    return os.environ["JIRA_BASE_URL"].rstrip("/")


def _auth() -> tuple[str, str]:
    return (os.environ["JIRA_EMAIL"], os.environ["JIRA_API_TOKEN"])


def _url(path: str) -> str:
    return f"{_base_url()}/rest/api/3/{path.lstrip('/')}"


def _request(method: str, path: str, **kwargs: Any) -> requests.Response:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    for attempt in range(GET_ATTEMPTS):
        try:
            resp = requests.request(
                method, _url(path), auth=_auth(), headers=headers, **kwargs
            )
            break
        except (requests.ConnectionError, requests.Timeout):
            if method.upper() != "GET" or attempt == GET_ATTEMPTS - 1:
                raise
            print(
                f"Jira API GET {path} connection failed; retrying in {RETRY_DELAY_SECONDS}s",
                file=sys.stderr,
            )
            time.sleep(RETRY_DELAY_SECONDS)
    if not resp.ok:
        print(
            f"Jira API {method} {path} failed: {resp.status_code} {resp.text}",
            file=sys.stderr,
        )
    return resp


def _text_runs_to_adf(text: str) -> list[dict]:
    """Split a line of text into ADF text nodes, turning bare URLs into
    clickable links via the "link" mark."""
    runs: list[dict] = []
    last_end = 0
    for match in URL_RE.finditer(text):
        if match.start() > last_end:
            runs.append({"type": "text", "text": text[last_end : match.start()]})
        url = match.group(0)
        runs.append(
            {
                "type": "text",
                "text": url,
                "marks": [{"type": "link", "attrs": {"href": url}}],
            }
        )
        last_end = match.end()
    if last_end < len(text):
        runs.append({"type": "text", "text": text[last_end:]})
    return runs


def text_to_adf(text: str) -> dict:
    """Wrap plain text in the Atlassian Document Format Jira requires,
    splitting on blank lines into paragraphs and auto-linking bare URLs."""
    paragraphs = text.split("\n\n") if text else [""]
    content = []
    for paragraph in paragraphs:
        runs = _text_runs_to_adf(paragraph)
        content.append({"type": "paragraph", "content": runs})
    return {"type": "doc", "version": 1, "content": content}


def adf_to_text(adf: dict | str | None) -> str:
    """Extract plain text from an Atlassian Document Format value."""
    if not adf:
        return ""
    if isinstance(adf, str):
        return adf

    def walk(node: dict) -> str:
        parts = []
        for child in node.get("content", []):
            if child.get("type") == "text":
                parts.append(child.get("text", ""))
            else:
                parts.append(walk(child))
        return "".join(parts)

    return "\n\n".join(walk(node) for node in adf.get("content", []))


def get_issue(key: str) -> dict | None:
    resp = _request("GET", f"issue/{key}")
    return resp.json() if resp.ok else None


def issue_is_missing(key: str) -> bool:
    resp = _request("GET", f"issue/{key}")
    return resp.status_code == 404


def update_description(key: str, description: str) -> bool:
    resp = _request(
        "PUT",
        f"issue/{key}",
        json={"fields": {"description": text_to_adf(description)}},
    )
    return resp.ok


def search(jql: str, fields: list[str], max_results: int = 100) -> list[dict]:
    issues: list[dict] = []
    next_page_token: str | None = None
    while True:
        params = {"jql": jql, "fields": ",".join(fields), "maxResults": max_results}
        if next_page_token:
            params["nextPageToken"] = next_page_token
        resp = _request("GET", "search/jql", params=params)
        if not resp.ok:
            resp.raise_for_status()
        data = resp.json()
        issues.extend(data.get("issues", []))
        next_page_token = data.get("nextPageToken")
        if data.get("isLast", True) or not next_page_token:
            break
    return issues


def list_comments(key: str) -> list[dict]:
    resp = _request("GET", f"issue/{key}/comment", params={"maxResults": 100})
    return resp.json().get("comments", []) if resp.ok else []
