"""Serving profile: the speed comparison times every model the way it would be served (bf16, its own fastest
engine) on one instance, without touching the evaluation path or the evaluation runs."""

import importlib.util
import math
import os
from pathlib import Path

import pytest

from decision_model.backends import get_backend
from decision_model.backends.base import Backend, label_key
from decision_model.config import CONFIG_DIR, load_model_config, load_yaml
from decision_model.data import write_jsonl
from decision_model.pipeline import execute, find_run

REPO = Path(__file__).resolve().parents[1]
COMPARED = ["gemma-4-26b-a4b", "kev-4b", "laya-multilingual", "chon-1.0-e2b", "chon-1.0-e4b"]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


entry = _load("entry", "sagemaker/entry.py")
launch = _load("launch", "scripts/launch_sagemaker.py")


# -- profiles ---------------------------------------------------------------------------

def test_every_compared_model_has_a_serving_profile_and_one_shared_instance():
    serving = load_yaml(CONFIG_DIR / "serving.yaml")
    assert serving["instance"] == "ml.g6e.xlarge"
    for name in COMPARED:
        config = load_model_config(name)
        assert config.get("serving"), name
        instance, volume = launch.job_settings(config, serving, None)
        assert instance == serving["instance"] and volume >= serving["volume_gb"]


def test_a_model_without_a_profile_cannot_run_a_serving_run():
    with pytest.raises(ValueError, match="no serving profile"):
        get_backend(load_model_config("clef-flash"), serving=True)


def test_kev_serves_bf16_with_graphs_and_fused_kernels_only_in_serving_runs():
    config = load_model_config("kev-4b")
    evaluation = get_backend(config).serve_env()
    assert evaluation == {"KEV_TEMPERATURE": "1.0", "KEV_DTYPE": "fp32"}
    serving = get_backend(config, serving=True).serve_env()
    assert serving == {"KEV_TEMPERATURE": "1.0", "KEV_DTYPE": "bf16", "KEV_CUDA_GRAPHS": "1", "KEV_FUSED": "1"}
    assert get_backend(config, serving=True).inference_precision().startswith("bf16")
    assert any(p.startswith("flash-linear-attention==") for p in config["serving"]["install"])


@pytest.mark.parametrize("name", ["chon-1.0-e2b", "chon-1.0-e4b"])
def test_chon_reads_through_vllm_only_in_serving_runs(name, tmp_path, monkeypatch):
    monkeypatch.setenv("CHON_REREAD", "0.5")
    config = load_model_config(name)
    evaluation, serving = get_backend(config), get_backend(config, serving=True)
    assert "--reader" not in evaluation.serve_command(8000)
    command = serving.serve_command(8000)
    assert command[command.index("--reader") + 1] == "vllm"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    # The same one-pass settings as the evaluation runs.
    assert command[command.index("--noul-commit") + 1] == "off" and "--no-think" in command

    from decision_model.backends.chon import clean_vllm_bin

    wrapper = clean_vllm_bin(venv=Path("/opt/vllm-venv"), directory=tmp_path)
    assert os.access(wrapper, os.X_OK)
    assert "unset PYTHONPATH" in wrapper.read_text() and "exec /opt/vllm-venv/bin/vllm" in wrapper.read_text()
    env = serving.serve_env(wrapper)
    assert env["CHON_VLLM_STRICT"] == "1" and env["CHON_VLLM_BIN"] == str(wrapper)
    assert "CHON_REREAD" not in env and "CHON_VLLM_STRICT" not in evaluation.serve_env()


def test_gemma_keeps_int8_weights_and_bf16_compute_on_the_l40s():
    """bf16 weights (~52 GB) do not fit the 48 GB L40S: the serving run uses the evaluation flags unchanged."""
    config = load_model_config("gemma-4-26b-a4b")
    evaluation, serving = get_backend(config).vllm_args(), get_backend(config, serving=True).vllm_args()
    assert serving == evaluation
    assert serving[serving.index("--dtype") + 1] == "bfloat16"
    assert serving[serving.index("--quantization") + 1] == "int8_per_channel_weight_only"


def test_laya_autocasts_to_bf16_only_in_serving_runs():
    config = load_model_config("laya-multilingual")
    assert get_backend(config).inference_precision() == "fp32 (amp off)"
    assert get_backend(config, serving=True).inference_precision().startswith("bf16")


# -- pipeline ---------------------------------------------------------------------------

def _keys(question):
    if question["type"] == "noul":
        return ["true", "false"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


class FakeBackend(Backend):
    def __init__(self):
        super().__init__({"backend": "fake"})
        self.checkpoints = []

    def train(self, *args, **kwargs):
        raise AssertionError("a serving run never trains")

    def predict(self, checkpoint, records):
        self.checkpoints.append(checkpoint)
        out = []
        for record in records:
            answers = {}
            for qid, q in record["questions"].items():
                keys = _keys(q)
                gold = label_key(q) if "label" in q else keys[0]
                answers[qid] = {k: math.log(0.9 if k == gold else 0.1 / (len(keys) - 1)) for k in keys}
            out.append({"id": record["id"], "answers": answers})
        return out


def _data(root):
    for lang in ("ko", "ja", "en"):
        for partition in ("train", "calibration", "development", "test"):
            rows = [{"id": f"{lang}:{partition}:{i}", "task": "s4", "lang": lang, "partition": partition,
                     "state": {"keyword": f"kw{i}"}, "label": "brand" if i % 2 == 0 else "nonbrand", "variant": None}
                    for i in range(4)]
            write_jsonl(root / "s4" / lang / f"{partition}.jsonl", rows)
    return root


def test_serving_run_reads_the_checkpoint_in_place_and_writes_beside_the_evaluation_run(tmp_path):
    data = _data(tmp_path / "data")
    checkpoint = tmp_path / "downloaded" / "checkpoint"
    checkpoint.mkdir(parents=True)
    backend = FakeBackend()
    summary = execute(find_run("kev-4b", "S-s4-both", 0), data, tmp_path / "runs", backend=backend,
                      measure_latency_too=False, checkpoint=checkpoint, serving=True)
    out = tmp_path / "runs" / "serving" / "kev-4b" / "S-s4-both" / "s0"
    assert (out / "summary.json").exists()
    assert not (tmp_path / "runs" / "kev-4b").exists()              # the evaluation run is untouched
    assert not (out / "checkpoint").exists()                         # not copied (or uploaded) again
    assert set(backend.checkpoints) == {checkpoint}
    assert summary["profile"] == "serving" and summary["serving"]["instance"] == "ml.g6e.xlarge"
    assert "train" not in summary["model_inputs_sha256"]


def test_serving_a_trained_run_needs_its_checkpoint(tmp_path):
    data = _data(tmp_path / "data")
    with pytest.raises(ValueError, match="needs its checkpoint"):
        execute(find_run("kev-4b", "S-s4-both", 0), data, tmp_path / "runs", backend=FakeBackend(),
                measure_latency_too=False, serving=True)
    # A run that does not train (B0) serves the released model.
    summary = execute(find_run("kev-4b", "B0-s4-noul", 0), data, tmp_path / "runs", backend=FakeBackend(),
                      measure_latency_too=False, serving=True)
    assert summary["profile"] == "serving"


# -- SageMaker job ------------------------------------------------------------------------

def test_serving_hyperparameter_becomes_a_bare_flag():
    assert entry.take_serving_flag(["--model", "m", "--serving", "1", "--runs", "G:0"]) == (
        ["--model", "m", "--runs", "G:0"], True)
    assert entry.take_serving_flag(["--model", "m", "--serving"]) == (["--model", "m"], True)
    assert entry.take_serving_flag(["--model", "m", "--serving", "false"]) == (["--model", "m"], False)
    assert entry.take_serving_flag(["--model", "m"]) == (["--model", "m"], False)


def test_checkpoint_path_is_read_from_the_run_summary():
    kev = {"checkpoint": "/opt/ml/model/kev-4b/S-s5-choice/s0/checkpoint/checkpoint"}
    assert entry.checkpoint_relative_path(kev, "kev-4b/S-s5-choice/s0") == "checkpoint/checkpoint"
    laya = {"checkpoint": "/opt/ml/model/laya-multilingual/S-s5-choice/s0/checkpoint"}
    assert entry.checkpoint_relative_path(laya, "laya-multilingual/S-s5-choice/s0") == "checkpoint"
    with pytest.raises(ValueError):
        entry.checkpoint_relative_path({"checkpoint": "jaredpalmer/kev-4b@v1.0"}, "kev-4b/B0-s5-choice/s0")


def test_serving_batch_passes_the_flag_and_uploads_under_serving(monkeypatch, tmp_path):
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
    calls, uploads = [], []
    monkeypatch.setattr(entry, "sh", lambda *command, **_: calls.append(command))
    monkeypatch.setattr(entry, "upload_outputs", lambda run_id=None: uploads.append(run_id))
    monkeypatch.setattr(entry, "serving_checkpoint_args",
                        lambda model, group, seed: [] if group.startswith("B0") else ["--checkpoint", f"/ck/{group}"])
    entry.run_batch("kev-4b", [("B0-s5-choice", 0), ("S-s5-choice", 0)], [], None, serving=True)
    assert all("--serving" in c for c in calls)
    assert "--checkpoint" not in calls[0] and calls[1][calls[1].index("--checkpoint") + 1] == "/ck/S-s5-choice"
    assert uploads == ["serving/kev-4b/B0-s5-choice/s0", "serving/kev-4b/S-s5-choice/s0"]


def test_launcher_names_and_places_serving_jobs():
    from types import SimpleNamespace as Run

    assert launch.output_id("kev-4b/S-s5-choice/s0", True) == "serving/kev-4b/S-s5-choice/s0"
    assert launch.output_id("kev-4b/S-s5-choice/s0", False) == "kev-4b/S-s5-choice/s0"
    runs = [Run(group="B0-s5-choice", seed=0), Run(group="S-s5-choice", seed=0)]
    assert launch.batch_job_name("kev-4b", runs, "serve") == "kev-4b-serve-b0s5ch-s5ch-s0"
    # Evaluation jobs keep the model's own instance.
    assert launch.job_settings(load_model_config("kev-4b"), None, None)[0] == "ml.g6e.xlarge"
