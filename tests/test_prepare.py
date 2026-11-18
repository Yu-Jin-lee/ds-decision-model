"""prepare.py on rows shaped like cdj_dataset/s5_master (field names and values
copied from the real master's schema, contents invented)."""

import gzip
import json

import pytest

from decision_model.config import CONFIG_DIR, load_yaml
from decision_model.data import to_model_record
from decision_model.formats import build_questions, load_task_specs
from decision_model.prepare import canonical, convert, read_master, serp_compact, serp_overview_or_top

TASKS = load_yaml(CONFIG_DIR / "tasks.yaml")
S5 = TASKS["tasks"]["s5"]
SPECS = load_task_specs(TASKS)


def model_state(record):
    return to_model_record(record, SPECS["s5"], "choice", with_labels=False)["state"]


def master_row(**changes):
    row = {
        "id": "kr:예시 무게", "lang": "ko", "partition": "train", "keyword": "예시 무게",
        "kind": "brand", "question": "cdj_brand", "stage": "browsing", "stage_source": "three_agree",
        "industry": "신발", "category": "러닝화", "target": ["예시"], "intent": ["무게"], "site": [],
        "ko": {"keyword": "", "target": "", "intent": "", "category": ""},
        "labels": {"rule": "browsing", "gemma": "browsing", "claude": "browsing", "human": None},
        "meta": {"type": "typical", "hard_type": None, "cue": "무게", "kind_changed": False},
        "context": {"job_id": "x", "volume": 93},
        "serp": {"features": [{"title": "t1", "snippet": "s1"}, {"title": "t2", "snippet": ""}],
                 "ai_overview": {"text": "overview", "sources": ["a"], "domains": ["a.com"]},
                 "serp_key": "abc"},
    }
    row.update(changes)
    return row


def test_s5_master_row_to_canonical_record():
    row = master_row()
    record = canonical(row, "s5", S5)
    assert record["label"] == "browsing" and record["variant"] == "brand"
    # Stored raw, so the SERP format can change without rebuilding the data.
    assert record["inputs"]["serp"] == row["serp"]
    assert "state" not in record
    state = model_state(record)
    assert state == {
        "keyword": "예시 무게", "target": ["예시"], "intent": ["무게"], "site": [],
        "industry": "신발", "category": "러닝화",
        "serp": {"ai_overview": "overview"},
        "brand": True,
    }
    assert record["meta"] == {"type": "typical", "hard_type": None, "stage_source": "three_agree", "kind_changed": False}
    # Nothing the model should not see: other labels, translations, job ids.
    assert "labels" not in json.dumps(state) and "job_id" not in json.dumps(state)
    assert "labels" not in json.dumps(record["inputs"]) and "job_id" not in json.dumps(record["inputs"])


def test_generic_row_maps_to_nonbrand_and_builds_questions():
    record = canonical(master_row(kind="generic", stage="own_and_service"), "s5", S5)
    assert record["variant"] == "nonbrand" and model_state(record)["brand"] is False
    questions = build_questions(load_task_specs(TASKS)["s5"], "both", record)
    assert set(questions["s5_choice"]["criteria"]) == {"initial_exploration", "own_and_service", "other"}
    assert questions["s5_score"]["label"] == 1


def test_label_must_fit_the_variant():
    with pytest.raises(ValueError, match="not allowed for brand"):
        canonical(master_row(stage="initial_exploration"), "s5", S5)
    with pytest.raises(ValueError, match="not in"):
        canonical(master_row(stage="own & service"), "s5", S5)


def test_convert_collects_errors_and_duplicates(tmp_path):
    rows = [master_row(), master_row(), master_row(id="kr:b", kind="unknown")]
    path = tmp_path / "m.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    records, errors = convert(read_master(path), "s5", S5)
    assert len(records) == 1
    assert any("duplicate" in e for e in errors) and any("kind='unknown'" in e for e in errors)


def test_serp_compact_handles_missing_parts():
    assert serp_compact({"features": [], "ai_overview": None}) == {}
    assert serp_compact(None) is None


def test_serp_overview_or_top():
    long = "x" * 1500
    serp = {"ai_overview": {"text": long, "sources": ["a"]},
            "features": [{"title": "t1", "snippet": "s1"}], "serp_key": "k"}
    assert serp_overview_or_top(serp) == {"ai_overview": "x" * 1000}
    no_overview = {"features": [{"title": f"t{i}", "snippet": f"s{i}"} for i in range(5)]}
    assert serp_overview_or_top(no_overview) == {
        "results": [{"title": "t0", "snippet": "s0"}, {"title": "t1", "snippet": "s1"}, {"title": "t2", "snippet": "s2"}]
    }
    assert serp_overview_or_top({"ai_overview": "  ", "features": []}) == {"results": []}


def test_serp_format_comes_from_config_at_load_time():
    record = canonical(master_row(), "s5", S5)
    raw = TASKS["tasks"]["s5"]
    changed = {**raw, "state": {**raw["state"], "serp": {"from": "serp", "format": "serp_compact"}}}
    spec = load_task_specs({"tasks": {"s5": changed}})["s5"]
    state = to_model_record(record, spec, "choice", with_labels=False)["state"]
    assert state["serp"] == {"ai_overview": "overview", "results": [{"title": "t1", "snippet": "s1"}, {"title": "t2"}]}
