"""Chon: released typed-decision models served over /v1/systemone, evaluated as is (B0)."""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

from decision_model.backends import get_backend
from decision_model.config import CONFIG_DIR, expand_runs, load_model_config, load_yaml
from decision_model.data import to_model_record
from decision_model.formats import load_task_specs

SPECS = load_task_specs(load_yaml(CONFIG_DIR / "tasks.yaml"))
REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["chon-1.0-e4b", "chon-1.0-e2b"])
def test_configs_serve_one_pass_without_commit_band(name):
    config = load_model_config(name)
    backend = get_backend(config)
    command = backend.serve_command(8123)
    assert command[:3] == [sys.executable, "-m", "chon.serve"]
    assert command[command.index("--model") + 1] == f"ninhnv/{name}"
    assert "--no-think" in command
    assert command[command.index("--noul-commit") + 1] == "off"
    assert command[command.index("--reread") + 1] == "off"
    assert command[command.index("--deadline") + 1] == "0"


def test_e4b_trains_e2b_stays_b0_only():
    """chon-1.0-e4b continues training its released adapter (tests/test_chon_train.py); e2b is still B0 only."""
    assert load_model_config("chon-1.0-e4b")["trainable"] is True
    assert load_model_config("chon-1.0-e4b")["train"]["lr"] == load_model_config("kev-4b")["train"]["lr"]
    assert load_model_config("chon-1.0-e2b")["trainable"] is False


def test_b0_formats_have_at_most_16_options():
    """Chon takes 2-16 options per choice question."""
    for task, spec in SPECS.items():
        for qid, question in spec.raw["questions"].items():
            variants = question.get("variants") or {None: question}
            for v in variants.values():
                assert len(v.get("criteria") or []) <= 16, (task, qid)


def test_predict_parses_chon_answers(monkeypatch):
    """Chon's answer objects (as chon.answers.format_answer builds them) become per-option log-probabilities."""
    from contextlib import contextmanager
    import math

    from decision_model.backends import chon

    backend = get_backend(load_model_config("chon-1.0-e2b"))
    s4 = {"id": "kr:1", "task": "s4", "lang": "ko", "state": {"keyword": "k"}, "label": "brand", "variant": None}
    s5 = {"id": "kr:1", "task": "s5", "lang": "ko", "state": {"keyword": "k"}, "label": "browsing", "variant": "brand"}
    records = [to_model_record(s4, SPECS["s4"], "noul", True, True), to_model_record(s5, SPECS["s5"], "score", True, True)]
    sent = []

    def fake_post(url, body):
        sent.append(body)
        answers = {}
        for qid, q in body["questions"].items():
            assert "label" not in q
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": 0.8, "probabilities": {"true": 0.8, "false": 0.2}}
            else:
                n = len(q["criteria"])
                answers[qid] = {"type": "score", "score": 1.0, "level": 0, "probabilities": {str(i): 1 / n for i in range(n)}}
        return {"answers": answers, "usage": {}}

    class FakeServer:
        returncode = None
        def poll(self): return None
        def terminate(self): pass
        def wait(self, timeout=None): return 0

    monkeypatch.setattr(chon.subprocess, "Popen", lambda *a, **k: FakeServer())
    monkeypatch.setattr(chon.ChonBackend, "_wait_ready", staticmethod(lambda *a: None))
    monkeypatch.setattr(chon.ChonBackend, "_post", staticmethod(fake_post))
    predictions = backend.predict(None, records)
    assert [p["id"] for p in predictions] == ["kr:1", "kr:1"]
    assert math.isclose(math.exp(predictions[0]["answers"]["s4_noul"]["true"]), 0.8)
    assert set(predictions[1]["answers"]["s5_score"]) == {"0", "1", "2", "3", "4"}
    started = []
    monkeypatch.setattr(chon.subprocess, "Popen", lambda command, **k: started.append(command) or FakeServer())
    backend.predict(Path("ckpt"), records[:1])   # a fine-tuned package is served in place of the release
    assert started[-1][started[-1].index("--model") + 1] == "ckpt"


def test_b0_batch_job_name_is_valid_for_sagemaker():
    spec = importlib.util.spec_from_file_location("launch", REPO / "scripts" / "launch_sagemaker.py")
    launch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launch)
    runs = expand_runs("chon-1.0-e4b", load_yaml(CONFIG_DIR / "groups.yaml"), load_yaml(CONFIG_DIR / "tasks.yaml"), groups={"B0"})
    runs = [r for r in runs if not r.group.endswith("-both")]
    assert len(runs) == 7
    name = launch.batch_job_name("chon-1.0-e4b", runs)
    assert re.fullmatch(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,38}", name), name
