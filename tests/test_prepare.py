"""prepare.py on rows shaped like cdj_dataset/s5_master (field names and values
copied from the real master's schema, contents invented)."""

import gzip
import json

import pytest

from decision_model.config import CONFIG_DIR, load_yaml
from decision_model.formats import build_questions, load_task_specs
from decision_model.prepare import canonical, convert, read_master, serp_compact

TASKS = load_yaml(CONFIG_DIR / "tasks.yaml")
S5 = TASKS["tasks"]["s5"]


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
    record = canonical(master_row(), "s5", S5)
    assert record["label"] == "browsing" and record["variant"] == "brand"
    assert record["state"] == {
        "keyword": "예시 무게", "target": ["예시"], "intent": ["무게"], "site": [],
        "industry": "신발", "category": "러닝화",
        "serp": {"ai_overview": "overview", "results": [{"title": "t1", "snippet": "s1"}, {"title": "t2"}]},
        "brand": True,
    }
    assert record["meta"] == {"type": "typical", "hard_type": None, "stage_source": "three_agree", "kind_changed": False}
    # Nothing the model should not see: other labels, translations, job ids.
    assert "labels" not in json.dumps(record["state"]) and "job_id" not in json.dumps(record["state"])


def test_generic_row_maps_to_nonbrand_and_builds_questions():
    record = canonical(master_row(kind="generic", stage="own_and_service"), "s5", S5)
    assert record["variant"] == "nonbrand" and record["state"]["brand"] is False
    questions = build_questions(load_task_specs(TASKS)["s5"], "both", record)
    assert set(questions["s5_choice"]["criteria"]) == {"initial_exploration", "own_and_service", "other"}
    assert questions["s5_stage"]["label"] == 1


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
