"""Multi-task runs with ids shared across tasks, and resuming a run from an existing checkpoint."""

import importlib.util
import io
import math
import tarfile
from pathlib import Path

import pytest

from decision_model.backends.base import Backend, label_key
from decision_model.config import CONFIG_DIR, load_yaml
from decision_model.data import to_model_record, write_jsonl
from decision_model.evaluation import calibration_items, paired, score
from decision_model.formats import load_task_specs
from decision_model.pipeline import execute, find_run

SPECS = load_task_specs(load_yaml(CONFIG_DIR / "tasks.yaml"))

spec = importlib.util.spec_from_file_location("entry", Path(__file__).resolve().parents[1] / "sagemaker" / "entry.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def _keys(question):
    if question["type"] == "noul":
        return ["true", "false"]
    if question["type"] == "choice":
        return list(question["criteria"])
    return [str(i) for i in range(len(question["criteria"]))]


def confident(record):
    """A prediction that puts 0.9 on each labelled question's gold option."""
    answers = {}
    for qid, question in record["questions"].items():
        keys = _keys(question)
        gold = label_key(question) if "label" in question else keys[0]
        answers[qid] = {k: math.log(0.9 if k == gold else 0.1 / (len(keys) - 1)) for k in keys}
    return {"id": record["id"], "answers": answers}


def shared_id_records():
    """An S4 and an S5 record built from the same keyword row: same id, different tasks."""
    s4 = {"id": "kr:1", "task": "s4", "lang": "ko", "state": {"keyword": "k"}, "label": "brand", "variant": None}
    s5 = {"id": "kr:1", "task": "s5", "lang": "ko", "state": {"keyword": "k"}, "label": "browsing", "variant": "brand"}
    return [
        to_model_record(s4, SPECS["s4"], "both", with_labels=True, for_prediction=True),
        to_model_record(s5, SPECS["s5"], "choice", with_labels=True, for_prediction=True),
    ]


def test_ids_shared_across_tasks_get_their_own_prediction():
    records = shared_id_records()
    predictions = [confident(r) for r in records]
    items = calibration_items(records, predictions)
    assert len(items) == 3  # s4_noul, s4_choice, s5_choice: none lost to the shared id
    rows, metrics = score(SPECS, records, predictions, 1.0)
    assert {(r["task"], r["view"]) for r in rows} == {("s4", "noul"), ("s4", "choice"), ("s5", "choice")}
    assert all(r["prediction"] == r["gold"] for r in rows)
    # Agreement compares the two S4 views only, not S4 with S5.
    assert metrics["s4/ko/agreement"] == {"n": 1, "agreement": 1.0}
    assert "s5/ko/agreement" not in metrics


def test_predictions_out_of_order_are_an_error():
    records = shared_id_records() + [dict(shared_id_records()[0], id="kr:2")]
    predictions = [confident(r) for r in records]
    with pytest.raises(ValueError, match="out of order"):
        paired(records, [predictions[0], predictions[2], predictions[1]])
    with pytest.raises(ValueError, match="predictions for"):
        paired(records, predictions[:2])


class ResumeBackend(Backend):
    def __init__(self):
        super().__init__({"backend": "fake", "checkpoint": "fake/base"})
        self.checkpoints = []

    def train(self, *args, **kwargs):
        raise AssertionError("training must be skipped when resuming")

    def predict(self, checkpoint, records):
        self.checkpoints.append(checkpoint)
        return [confident(r) for r in records]


def test_resume_skips_training_and_keeps_the_checkpoint_with_the_run(tmp_path):
    data = tmp_path / "data"
    for lang in ("ko", "ja", "en"):
        for partition in ("train", "calibration", "development", "test"):
            rows = [{"id": f"{lang}:{partition}:{i}", "task": "s4", "lang": lang, "partition": partition,
                     "state": {"keyword": f"kw{i}"}, "label": "brand" if i % 2 == 0 else "nonbrand", "variant": None}
                    for i in range(4)]
            write_jsonl(data / "s4" / lang / f"{partition}.jsonl", rows)
    earlier = tmp_path / "earlier" / "checkpoint"
    earlier.mkdir(parents=True)
    (earlier / "adapter_model.safetensors").write_text("weights")

    backend = ResumeBackend()
    summary = execute(find_run("kev-4b", "S-s4-both", 0), data, tmp_path / "runs", backend=backend,
                      measure_latency_too=False, checkpoint=earlier)
    out = tmp_path / "runs" / "kev-4b" / "S-s4-both" / "s0"
    local = out / "checkpoint" / "checkpoint"
    assert (local / "adapter_model.safetensors").read_text() == "weights"
    assert set(backend.checkpoints) == {local}
    assert summary["resumed_from"] == str(earlier) and summary["checkpoint"] == str(local)
    assert "train" in summary["model_inputs_sha256"] and (out / "summary.json").exists()

    with pytest.raises(FileNotFoundError):
        execute(find_run("kev-4b", "S-s4-both", 0), data, tmp_path / "runs2", backend=backend,
                measure_latency_too=False, checkpoint=tmp_path / "missing")


def test_entry_resumes_runs_whose_checkpoint_is_in_the_earlier_job(monkeypatch, tmp_path):
    archive = tmp_path / "model.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("kev-4b/J-best/s0/checkpoint/checkpoint/head.pt")
        info.size = 4
        tar.addfile(info, io.BytesIO(b"head"))
    monkeypatch.setenv("RESUME_DIR", str(tmp_path / "resume"))
    root = entry.fetch_resume(str(archive))
    assert entry.resume_args(root, "kev-4b/J-best/s0") == [
        "--checkpoint", str(root / "kev-4b" / "J-best" / "s0" / "checkpoint" / "checkpoint")]
    assert entry.resume_args(root, "kev-4b/J-best/s1") == []
    assert entry.resume_args(None, "kev-4b/J-best/s0") == []

    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path / "model"))
    calls = []
    monkeypatch.setattr(entry, "sh", lambda *command, **_: calls.append(command))
    monkeypatch.setattr(entry, "upload_outputs", lambda run_id=None: None)
    entry.run_batch("kev-4b", [("J-best", 0), ("J-best", 1)], [], root)
    assert "--checkpoint" in calls[0] and "--checkpoint" not in calls[1]


def test_resume_flag_is_not_passed_on_to_run_py():
    assert entry._drop(["--model", "m", "--resume-from", "s3://b/k", "--runs", "G:0"], "--resume-from") == [
        "--model", "m", "--runs", "G:0"]
