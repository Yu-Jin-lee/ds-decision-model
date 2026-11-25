import math

import pytest

from decision_model.backends import get_backend
from decision_model.backends.prompt_llm import fill, one_hot_log_probs, parse_answer, prompt_fields, render_serp
from decision_model.formats import INVALID
from decision_model.config import load_model_config


@pytest.fixture(scope="module")
def backend():
    return get_backend(load_model_config("gemma-4-26b-a4b"))


def test_every_task_variant_language_has_a_prompt(backend):
    cases = [("s5", "brand"), ("s5", "nonbrand"), ("s4", None), ("s2", None)]
    for task, variant in cases:
        spec = backend.prompt_spec(task, variant)
        for lang in ("ko", "ja", "en"):
            assert "# Input" in backend.template(spec["file"], lang)


def test_digits_cover_the_task_labels(backend):
    from decision_model.config import load_yaml, CONFIG_DIR

    tasks = load_yaml(CONFIG_DIR / "tasks.yaml")["tasks"]
    assert set(backend.prompt_spec("s5", "brand")["digits"].values()) == set(tasks["s5"]["variant_labels"]["brand"])
    assert set(backend.prompt_spec("s5", "nonbrand")["digits"].values()) == set(tasks["s5"]["variant_labels"]["nonbrand"])
    assert set(backend.prompt_spec("s4", None)["digits"].values()) == set(tasks["s4"]["final_labels"])
    assert set(backend.prompt_spec("s2", None)["digits"].values()) == set(tasks["s2"]["final_labels"])


def test_s2_prompt_is_filled_and_keeps_literal_braces(backend):
    record = {"id": "x", "task": "s2", "lang": "ko", "variant": None,
              "state": {"industry": "생활용품", "category": "치약", "seeds": ["페리오", "2080"],
                        "keyword": "페리오 주가", "serp": {"ai_overview": "주가 정보"}}}
    prompt, digits = backend.build_prompt(record)
    assert "market category: 생활용품 > 치약" in prompt
    assert "related entities: 페리오, 2080" in prompt
    assert "keyword: 페리오 주가" in prompt and "search results: 주가 정보" in prompt
    assert "{major_category} > {minor_category}" in prompt   # text in the instructions, untouched
    assert "{keyword}" not in prompt and digits["4"] == "stock_finance"


def test_s5_uses_the_variant_and_language_prompt(backend):
    record = {"id": "x", "task": "s5", "lang": "ja", "variant": "nonbrand",
              "state": {"keyword": "ロボット掃除機 おすすめ", "target": "ロボット掃除機", "intent": "おすすめ", "site": None,
                        "serp": {"results": [{"title": "t", "snippet": "s"}]}}}
    prompt, digits = backend.build_prompt(record)
    assert "Choose among the 3 stages" in prompt and "おすすめ" in prompt
    assert "site: \n" in prompt and "- t: s" in prompt
    assert digits == {"1": "initial_exploration", "2": "own_and_service", "3": "other"}


@pytest.mark.parametrize("text,label", [
    ("2", "experience"), (" 2\n", "experience"), ("**4**", "own_and_service"), ("5", "other"),
    ("6", INVALID), ("Browsing", INVALID), ("", INVALID),
])
def test_parse_s5_brand_answers(backend, text, label):
    assert parse_answer(text, backend.prompt_spec("s5", "brand")["digits"]) == label


@pytest.mark.parametrize("text,label", [
    ("0", "related"), ("1|샴푸", "other_category"), ("2|大学", "same_name_other_category"),
    ("7|factory tour", "other_unrelated"), ("3", "hiring"), ("9", INVALID), ("type_num=4", INVALID),
])
def test_parse_s2_answers(backend, text, label):
    assert parse_answer(text, backend.prompt_spec("s2", None)["digits"]) == label


def test_parse_s4_answers(backend):
    digits = backend.prompt_spec("s4", None)["digits"]
    assert parse_answer("1|samsung", digits) == "brand"
    assert parse_answer("0|", digits) == "nonbrand"
    assert parse_answer("1|나르왈, 에코백스", digits) == "brand"


def test_one_hot_keeps_invalid_as_its_own_label():
    out = one_hot_log_probs(INVALID, ["a", "b"])
    assert out[INVALID] == 0.0 and out["a"] < -20 and out["b"] < -20


def test_only_the_prompt_level_is_supported(backend):
    with pytest.raises(ValueError, match="P0"):
        backend.generated_question({"id": "x", "questions": {"s5_choice": {"type": "choice"}}})
    assert backend.generated_question({"id": "x", "questions": {"s5_prompt": {"type": "generated"}}}) == "s5_prompt"
    with pytest.raises(NotImplementedError):
        backend.train([], None, 0)


def test_generated_answers_score_accuracy_and_invalid_as_wrong():
    from decision_model.config import CONFIG_DIR, load_yaml
    from decision_model.evaluation import score
    from decision_model.formats import build_questions, load_task_specs

    specs = load_task_specs(load_yaml(CONFIG_DIR / "tasks.yaml"))
    records = []
    for i, gold in enumerate(["experience", "browsing", "other"]):
        r = {"id": str(i), "task": "s5", "lang": "ko", "variant": "brand", "label": gold, "gold": gold, "level": "prompt"}
        r["questions"] = build_questions(specs["s5"], "prompt", r)
        assert r["questions"] == {"s5_prompt": {"type": "generated", "label": gold}}
        records.append(r)
    labels = ["browsing", "experience", "confirmation", "own_and_service", "other"]
    parsed = ["experience", "experience", INVALID]
    predictions = [{"id": str(i), "answers": {"s5_prompt": one_hot_log_probs(p, labels)}} for i, p in enumerate(parsed)]
    rows, metrics = score(specs, records, predictions, 1.0)
    cell = metrics["s5/ko/prompt"]
    assert cell["n"] == 3 and abs(cell["accuracy"] - 1 / 3) < 1e-9 and abs(cell["invalid_rate"] - 1 / 3) < 1e-9
    assert cell["brier"] is None and cell["ece"] is None
    assert [r["prediction"] for r in rows] == parsed


def test_b0_sweep_skips_prompt_and_p0_groups_are_explicit():
    from decision_model.config import CONFIG_DIR, expand_runs, load_yaml

    groups, tasks = load_yaml(CONFIG_DIR / "groups.yaml"), load_yaml(CONFIG_DIR / "tasks.yaml")
    b0 = expand_runs("kev-4b", groups, tasks, groups={"B0"})
    assert b0 and all("prompt" not in r.group for r in b0)
    assert not [r for r in expand_runs("kev-4b", groups, tasks, stages={0}) if r.group.startswith("P0")]   # optional
    p0 = expand_runs("gemma-4-26b-a4b", groups, tasks, groups={"P0-s2", "P0-s4", "P0-s5"})
    assert [(r.group, r.levels, r.train) for r in p0] == [
        ("P0-s2", {"s2": "prompt"}, False), ("P0-s4", {"s4": "prompt"}, False), ("P0-s5", {"s5": "prompt"}, False)]


def test_render_serp_and_fill():
    assert render_serp(None) == "" and render_serp({"ai_overview": "A"}) == "A"
    assert fill("{a} {b} {c}", {"a": "1", "b": "2"}) == "1 2 {c}"
    assert prompt_fields("s4", {"keyword": "k", "target": None, "serp": None}) == {"keyword": "k", "target": "", "serp": ""}


def test_p0_pipeline_end_to_end_with_a_fake_server(tmp_path, monkeypatch):
    """Full run: production prompt -> generated text -> parsed label -> metrics and latency."""
    from contextlib import contextmanager

    from decision_model.backends import prompt_llm
    from decision_model.data import write_jsonl
    from decision_model.pipeline import execute, find_run

    data = tmp_path / "data"
    for lang in ("ko", "ja", "en"):
        for partition in ("train", "calibration", "development", "test"):
            rows = [{"id": f"{lang}:{partition}:{i}", "task": "s4", "lang": lang, "partition": partition,
                     "state": {"keyword": f"kw{i}", "target": "brand" if i % 2 == 0 else "generic", "serp": None},
                     "label": "brand" if i % 2 == 0 else "nonbrand", "variant": None} for i in range(6)]
            write_jsonl(data / "s4" / lang / f"{partition}.jsonl", rows)

    backend = get_backend(load_model_config("gemma-4-26b-a4b"))
    sent = []

    @contextmanager
    def fake_server(self, checkpoint):
        def answer(record):
            prompt, digits = self.build_prompt(record)
            sent.append(prompt)
            target = record["state"]["target"]
            text = "1|" + target if target == "brand" else ("0|" if record["lang"] != "en" else "generic")
            label = prompt_llm.parse_answer(text, digits)
            qid = self.generated_question(record)
            return {"id": record["id"], "answers": {qid: prompt_llm.one_hot_log_probs(label, sorted(set(digits.values())))},
                    "output": text, "parsed": label}
        yield answer

    monkeypatch.setattr(prompt_llm.PromptLLMBackend, "predictor", fake_server)
    monkeypatch.setattr(prompt_llm.PromptLLMBackend, "vllm_version", lambda self: "test")
    summary = execute(find_run("gemma-4-26b-a4b", "P0-s4", 0), data, tmp_path / "runs", backend=backend)
    cells = summary["metrics_development"]
    assert cells["s4/ko/prompt"]["accuracy"] == 1.0 and cells["s4/ko/prompt"]["invalid_rate"] == 0.0
    assert cells["s4/en/prompt"]["accuracy"] == 0.5 and cells["s4/en/prompt"]["invalid_rate"] == 0.5
    assert summary["temperature"] == 1.0 and "calibration" not in summary["model_inputs_sha256"]
    assert summary["latency_development"]["records"] == 18
    assert any("[Target]\nbrand" in p for p in sent)
