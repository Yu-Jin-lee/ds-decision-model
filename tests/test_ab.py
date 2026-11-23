"""A/B check on outside data (scripts/jf_to_master.py, scripts/ab_job.py) and the serving batch's disk cleanup."""

from __future__ import annotations

import csv
import io
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "sagemaker"))

FIELDS = ["country", "job_id", "keyword", "industry", "category", "brand", "target", "intent", "site", "serp",
          "serp_type", "serp_source", "serp_requested_at", "run_serp_query", "cdj_stage", "stage_num"]


def write_jf_csv(path: Path) -> None:
    rows = [
        {"country": "kr", "job_id": "j", "keyword": "남자 올영 추천", "industry": "뷰티/화장품", "category": "남성 기초",
         "brand": "False", "target": "", "intent": "남자, 추천", "site": "올영",
         "serp": json.dumps({"ai_overview": "개요"}, ensure_ascii=False), "serp_type": "ai_overview",
         "cdj_stage": "initial_exploration", "stage_num": "1"},
        {"country": "kr", "job_id": "j", "keyword": "토리든 후기", "industry": "뷰티/화장품", "category": "남성 기초",
         "brand": "True", "target": "토너", "intent": "후기", "site": "",
         "serp": json.dumps({"results": [{"title": "t", "snippet": "s"}]}, ensure_ascii=False),
         "serp_type": "results", "cdj_stage": "experience", "stage_num": "2"},
        {"country": "us", "job_id": "k", "keyword": "retinol", "industry": "beauty", "category": "anti-aging",
         "brand": "False", "target": "", "intent": "", "site": "", "serp": "", "serp_type": "",
         "cdj_stage": "other", "stage_num": "3"},
    ]
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)


def prepare(tmp_path: Path) -> Path:
    write_jf_csv(tmp_path / "s5.csv")
    master, data = tmp_path / "m.jsonl.gz", tmp_path / "data"
    subprocess.run([sys.executable, str(REPO / "scripts" / "jf_to_master.py"), "--task", "s5", "--csv",
                    str(tmp_path / "s5.csv"), "--country", "kr", "--out", str(master)], check=True)
    subprocess.run([sys.executable, str(REPO / "scripts" / "prepare_data.py"), "--task", "s5", "--master",
                    str(master), "--out", str(data)], check=True)
    return data


def test_jf_rows_become_the_training_input_format(tmp_path):
    from decision_model.config import CONFIG_DIR, load_yaml
    from decision_model.data import build_records
    from decision_model.formats import load_task_specs

    data = prepare(tmp_path)
    specs = load_task_specs(load_yaml(CONFIG_DIR / "tasks.yaml"))
    records = build_records(data, specs, {"s5": "choice"}, ["ko"], "development", with_labels=True,
                            for_prediction=True)
    assert [r["id"] for r in records] == ["kr:남자 올영 추천", "kr:토리든 후기"]   # country filter
    first, second = records
    assert first["state"]["intent"] == ["남자", "추천"] and first["state"]["target"] == []
    assert first["state"]["serp"] == {"ai_overview": "개요"}
    assert second["state"]["serp"] == {"results": [{"title": "t", "snippet": "s"}]}
    assert (first["variant"], second["variant"]) == ("nonbrand", "brand")
    assert (first["gold"], second["gold"]) == ("initial_exploration", "experience")
    assert list(first["questions"]) == ["s5_choice"]


class FakeS3:
    def __init__(self, summary):
        self.summary, self.uploaded = summary, []

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(json.dumps(self.summary).encode())}

    def get_paginator(self, name):
        class Pages:
            def paginate(self, **kwargs):
                return [{"Contents": [{"Key": f"{kwargs['Prefix']}adapter_model.safetensors"}]}]
        return Pages()

    def download_file(self, bucket, key, target):
        Path(target).write_text("x")


class FakeBackend:
    def __init__(self):
        self.checkpoint = None

    def predict(self, checkpoint, records):
        self.checkpoint = checkpoint
        out = []
        for i, record in enumerate(records):
            options = list(record["questions"]["s5_choice"]["criteria"])
            answer = record["gold"] if i == 0 else options[0]
            out.append({"id": record["id"], "answers": {"s5_choice": {o: (0.9 if o == answer else 0.1 / (len(options) - 1))
                                                                     for o in options}}})
        return out

    def inference_precision(self):
        return "fake"


def test_ab_job_scores_with_the_runs_checkpoint_and_temperature(tmp_path, monkeypatch):
    import boto3

    import ab_job
    import decision_model.backends

    data = prepare(tmp_path)
    summary = {"temperature": 1.4, "git_commit": "abc", "metrics_development": {},
               "checkpoint": "/opt/ml/model/chon-1.0-e4b/J-best/s0/checkpoint/checkpoint"}
    backend = FakeBackend()
    monkeypatch.setattr(boto3, "client", lambda *a, **k: FakeS3(summary))
    monkeypatch.setattr(decision_model.backends, "get_backend", lambda config: backend)
    monkeypatch.setenv("AB_CHECKPOINT_DIR", str(tmp_path / "ck"))
    monkeypatch.delenv("SM_MODEL_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["ab_job.py", "--model", "chon-1.0-e4b", "--group", "J-best", "--task", "s5",
                                      "--lang", "ko", "--name", "t", "--data-root", str(data), "--runs-prefix",
                                      "s3://b/p/runs/", "--out", str(tmp_path / "out"), "--install", "0"])
    ab_job.main()

    out = tmp_path / "out" / "ab" / "t" / "chon-1.0-e4b" / "J-best" / "s0"
    result = json.loads((out / "summary.json").read_text())
    assert result["level"] == "choice" and result["records"] == 2
    assert result["source_run"]["temperature"] == 1.4
    assert backend.checkpoint == tmp_path / "ck" / "chon-1.0-e4b" / "J-best" / "s0" / "checkpoint" / "checkpoint"
    assert result["metrics_development"]["s5/ko/choice"]["accuracy"] == 0.5
    with open(out / "keywords.csv", encoding="utf-8-sig") as handle:
        lines = list(csv.DictReader(handle))
    assert [(r["keyword"], r["match"]) for r in lines] == [("남자 올영 추천", "1"), ("토리든 후기", "0")]


def test_serving_batch_frees_a_trained_runs_checkpoint_and_merged_copy(tmp_path, monkeypatch):
    import entry

    merged, checkpoints = tmp_path / "merged", tmp_path / "ck"
    monkeypatch.setenv("CHON_MERGED_DIR", str(merged))
    monkeypatch.setenv("SERVING_CHECKPOINTS_DIR", str(checkpoints))
    (merged / "release").mkdir(parents=True)
    before = entry._merged_entries()
    (merged / "trained").mkdir()
    (merged / "trained.tmp").mkdir()
    (checkpoints / "run").mkdir(parents=True)

    entry.free_serving_disk([], before)          # B0: nothing downloaded, nothing removed
    assert (merged / "trained").exists()
    entry.free_serving_disk(["--checkpoint", str(checkpoints / "run")], before)
    assert sorted(p.name for p in merged.iterdir()) == ["release"]
    assert not checkpoints.exists()


def test_vllm_env_turns_off_the_flashinfer_sampler(monkeypatch):
    from decision_model.backends.prompt_llm import _clean_env

    monkeypatch.delenv("VLLM_USE_FLASHINFER_SAMPLER", raising=False)
    assert _clean_env()["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
