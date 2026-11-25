"""Kev runs in its own uv environment; the host image's Python and uv settings must not leak into it."""

from decision_model.backends.kev import uv_environment


def test_uv_environment_isolates_kev_from_host_python(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/usr/lib64/python3.12")
    monkeypatch.setenv("PYTHONHOME", "/usr")
    monkeypatch.setenv("VIRTUAL_ENV", "/opt/venv")
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/opt/venv")
    monkeypatch.setenv("UV_PYTHON", "/opt/venv/bin/python")
    monkeypatch.setenv("HF_HOME", "/cache")
    env = uv_environment("/opt/kev", KEV_DTYPE="fp32")
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PYTHON"} & set(env)
    # uv must build Kev's own venv, never the image's /opt/venv.
    assert env["UV_PROJECT_ENVIRONMENT"] == "/opt/kev/.venv"
    assert env["HF_HOME"] == "/cache" and env["KEV_DTYPE"] == "fp32"


def test_train_stops_when_kev_drops_records(tmp_path):
    import sys
    import pytest
    from decision_model.backends.kev import TrainingDataDropped, run_train_without_drops

    script = tmp_path / "fake_train.py"
    script.write_text(
        "import time\n"
        "print('dropped 3 of 10 records that exceed the training context (384 state / 1024 branch / 2048 packed tokens)', flush=True)\n"
        "time.sleep(30)\n"
    )
    with pytest.raises(TrainingDataDropped, match="dropped 3 of 10"):
        run_train_without_drops([sys.executable, str(script)])


def test_train_passes_when_nothing_dropped(tmp_path):
    import sys
    from decision_model.backends.kev import run_train_without_drops

    script = tmp_path / "fake_train.py"
    script.write_text("print('2083 training requests', flush=True)\n")
    run_train_without_drops([sys.executable, str(script)])


def test_train_command_passes_max_state():
    from decision_model.backends.kev import KevBackend
    import inspect
    assert "--max_state" in inspect.getsource(KevBackend.train_command)
