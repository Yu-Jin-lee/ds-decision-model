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
