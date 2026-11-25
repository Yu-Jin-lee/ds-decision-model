from decision_model import pipeline


def test_commit_and_dirty_come_from_launcher_env(monkeypatch):
    monkeypatch.setenv("DECISION_MODEL_GIT_COMMIT", "abc1234")
    monkeypatch.setenv("DECISION_MODEL_GIT_DIRTY", "0")
    assert pipeline._git_commit() == "abc1234"
    assert pipeline._git_dirty() is False
    monkeypatch.setenv("DECISION_MODEL_GIT_DIRTY", "1")
    assert pipeline._git_dirty() is True
