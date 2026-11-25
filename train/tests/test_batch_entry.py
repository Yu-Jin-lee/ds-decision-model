"""Batch jobs: one SageMaker job runs several (group, seed) cells in sequence."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("entry", Path(__file__).resolve().parents[1] / "sagemaker" / "entry.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_parse_runs():
    assert entry.parse_runs("B0-s5-score:0,S-s5-choice:0, J+R-all:2") == [
        ("B0-s5-score", 0), ("S-s5-choice", 0), ("J+R-all", 2)]
    with pytest.raises(ValueError):
        entry.parse_runs("S-s5-choice")


def test_batch_keeps_only_extra_flags():
    args = ["--model", "kev-4b", "--runs", "S-s5-choice:0", "--final"]
    assert entry._without(args, "--runs") == ["--final"]


def test_batch_runs_each_cell_in_its_own_process_and_continues_after_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("SM_MODEL_DIR", str(tmp_path))
    calls, uploads = [], []

    def fake_sh(*command, **_):
        calls.append(command)
        if "S-s5-score" in command:
            raise entry.subprocess.CalledProcessError(1, command)
        group, seed = command[command.index("--group") + 1], command[command.index("--seed") + 1]
        (tmp_path / "m" / group / f"s{seed}").mkdir(parents=True)

    monkeypatch.setattr(entry, "sh", fake_sh)
    monkeypatch.setattr(entry, "upload_outputs", lambda run_id=None: uploads.append(run_id))
    with pytest.raises(SystemExit):
        entry.run_batch("m", [("S-s5-choice", 0), ("S-s5-score", 0), ("S-s5-both", 0)], [])
    assert len(calls) == 3 and all("run.py" in c[1] for c in calls)  # one process per run, all attempted
    assert uploads == ["m/S-s5-choice/s0", "m/S-s5-both/s0"]          # failed run not uploaded
    assert not (tmp_path / "m" / "S-s5-choice" / "s0").exists()       # local copy removed after upload
