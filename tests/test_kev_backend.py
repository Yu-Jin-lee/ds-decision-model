"""Kev runs in its own uv environment; the host image's Python paths must not leak into it."""

from decision_model.backends.kev import uv_environment


def test_uv_environment_drops_host_python_paths(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/usr/lib64/python3.12")
    monkeypatch.setenv("PYTHONHOME", "/usr")
    monkeypatch.setenv("VIRTUAL_ENV", "/opt/venv")
    monkeypatch.setenv("HF_HOME", "/cache")
    env = uv_environment(KEV_DTYPE="fp32")
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"} & set(env)
    assert env["HF_HOME"] == "/cache" and env["KEV_DTYPE"] == "fp32"
