"""Tests for deltares_ai_platform_sdk.optuna_utils."""

import pytest


def test_numbered_studies_are_new_unless_one_is_resumed(optuna_storage: str) -> None:
    optuna = pytest.importorskip("optuna")
    from deltares_ai_platform_sdk import optuna_utils

    storage = optuna_utils.storage_url()
    assert storage == optuna_storage

    first = optuna_utils.create_numbered_study(storage, "project-abc")
    assert first.study_name == "project-abc-001"
    assert optuna_utils.best_completed_value(first) is None
    for value in (0.5, 0.2):
        first.tell(first.ask(), value)
    first.tell(first.ask(), state=optuna.trial.TrialState.FAIL)

    second = optuna_utils.create_numbered_study(storage, "project-abc")
    assert second.study_name == "project-abc-002"
    assert optuna_utils.best_completed_value(second) is None

    existing = optuna_utils.list_studies(storage, "project-abc")
    assert sorted(existing) == ["project-abc-001", "project-abc-002"]
    assert existing["project-abc-001"].n_trials == 3
    resumed = optuna_utils.load_study(storage, "project-abc-001")
    assert optuna_utils.best_completed_value(resumed) == 0.2


@pytest.mark.parametrize("name", ["not-a-study", "other-0a1b-001", "project-0a1b"])
def test_parse_study_name_rejects_malformed_names(name: str) -> None:
    from deltares_ai_platform_sdk import optuna_utils

    assert optuna_utils.parse_study_name("project-0a1b-007", "project") == ("0a1b", 7)
    with pytest.raises(ValueError):
        optuna_utils.parse_study_name(name, "project")
