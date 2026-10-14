"""Minimal GitHub REST API client used by the epic-sync workflows."""

from __future__ import annotations

import os
import sys
from typing import Any

import requests

API_BASE = "https://api.github.com"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _request(method: str, path: str, **kwargs: Any) -> requests.Response:
    repo = os.environ["GITHUB_REPOSITORY"]
    resp = requests.request(
        method, f"{API_BASE}/repos/{repo}{path}", headers=_headers(), **kwargs
    )
    if not resp.ok:
        print(
            f"GitHub API {method} {path} failed: {resp.status_code} {resp.text}",
            file=sys.stderr,
        )
    return resp


def get_issue(number: int) -> dict | None:
    resp = _request("GET", f"/issues/{number}")
    return resp.json() if resp.ok else None


def create_issue(title: str, body: str) -> int | None:
    resp = _request("POST", "/issues", json={"title": title, "body": body})
    return resp.json()["number"] if resp.ok else None


def list_comments(number: int) -> list[dict]:
    resp = _request("GET", f"/issues/{number}/comments", params={"per_page": 100})
    return resp.json() if resp.ok else []


def add_comment(number: int, body: str) -> bool:
    resp = _request("POST", f"/issues/{number}/comments", json={"body": body})
    return resp.ok


def set_state(number: int, state: str) -> bool:
    """state must be 'open' or 'closed'."""
    resp = _request("PATCH", f"/issues/{number}", json={"state": state})
    return resp.ok


def update_issue(number: int, **fields: Any) -> bool:
    resp = _request("PATCH", f"/issues/{number}", json=fields)
    return resp.ok


def list_sub_issue_ids(parent_number: int) -> set[int]:
    resp = _request("GET", f"/issues/{parent_number}/sub_issues")
    if not resp.ok:
        return set()
    data = resp.json()
    sub_issues = data if isinstance(data, list) else data.get("sub_issues", [])
    return {issue["id"] for issue in sub_issues}


def add_sub_issue(parent_number: int, sub_issue_id: int) -> bool:
    resp = _request(
        "POST",
        f"/issues/{parent_number}/sub_issues",
        json={"sub_issue_id": sub_issue_id},
    )
    return resp.ok


def search_issues(query: str) -> list[dict]:
    """Search issues/PRs across the whole repo (not limited to one item)."""
    items: list[dict] = []
    page = 1
    while True:
        resp = requests.request(
            "GET",
            f"{API_BASE}/search/issues",
            headers=_headers(),
            params={"q": query, "per_page": 100, "page": page},
        )
        if not resp.ok:
            print(
                f"GitHub API GET /search/issues failed: {resp.status_code} {resp.text}",
                file=sys.stderr,
            )
            break
        page_items = resp.json().get("items", [])
        items.extend(page_items)
        if len(page_items) < 100:
            break
        page += 1
    return items
