"""GitHub Actions helpers that keep GitHub issues/PRs and a Jira epic in sync."""

from .epic_sync import main

__all__ = ["main"]
