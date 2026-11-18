import copy

import numpy as np
import pytest

from decision_model.calibration import apply_temperature, fit_temperature
from decision_model.config import CONFIG_DIR, expand_runs, load_yaml
from decision_model.formats import build_questions, final_distributions, load_task_specs
from decision_model.metrics import auto_rate, paired_bootstrap, summarize

TASKS = load_yaml(CONFIG_DIR / "tasks.yaml")
GROUPS = load_yaml(CONFIG_DIR / "groups.yaml")
SPECS = load_task_specs(TASKS)


def record(task, label, variant=None):
    return {"id": "x", "task": task, "lang": "ko", "state": {"keyword": "k"}, "label": label, "variant": variant}


# ----- formats -----------------------------------------------------------------

def test_s4_levels_build_expected_questions():
    r = record("s4", "brand")
    assert build_questions(SPECS["s4"], "noul", r) == {
        "s4_noul": {"type": "noul", "instructions": SPECS["s4"].raw["questions"]["s4_noul"]["instructions"], "label": True}
    }
    both = build_questions(SPECS["s4"], "both", r)
    assert set(both) == {"s4_noul", "s4_choice"}
    assert both["s4_choice"]["label"] == "brand"
    assert build_questions(SPECS["s4"], "score", record("s4", "nonbrand"))["s4_score"]["label"] == 0


def test_gate_inner_question_only_on_non_gate_records():
    related = build_questions(SPECS["s2"], "split", record("s2", "related"))
    assert related == {"s2_related": {**related["s2_related"], "label": True}}
    unrelated = build_questions(SPECS["s2"], "split", record("s2", "hiring"))
    assert unrelated["s2_related"]["label"] is False
    assert unrelated["s2_type"]["label"] == "hiring"
    # At prediction time both are always asked.
    assert set(build_questions(SPECS["s2"], "split", record("s2", "related"), with_labels=False)) == {"s2_related", "s2_type"}


def test_s5_variant_selects_options_and_levels():
    brand = build_questions(SPECS["s5"], "both", record("s5", "confirmation", "brand"))
    assert "browsing" in brand["s5_choice"]["criteria"]
    assert brand["s5_score"]["label"] == 2
    assert build_questions(SPECS["s5"], "score", record("s5", "other", "brand"))["s5_score"]["label"] == 4
    nonbrand = build_questions(SPECS["s5"], "score", record("s5", "own_and_service", "nonbrand"))
    assert nonbrand["s5_score"]["criteria"][nonbrand["s5_score"]["label"]].startswith("Own")
    assert build_questions(SPECS["s5"], "score", record("s5", "other", "nonbrand"))["s5_score"]["label"] == 2
    with pytest.raises(ValueError):
        build_questions(SPECS["s5"], "choice", record("s5", "initial_exploration", "brand"))


def test_gate_distribution_combines_and_sums_to_one():
    # s2 split: gate s2_related (true -> related), inner s2_type over the unrelated types.
    inner_keys = list(SPECS["s2"].question("s2_type", None)["criteria"])
    answers = {"s2_related": {"true": 0.2, "false": 0.8}, "s2_type": {k: (1.0 if k == "hiring" else 0.0) for k in inner_keys}}
    split = final_distributions(SPECS["s2"], "split", record("s2", "hiring"), answers)["split"]
    assert split["related"] == pytest.approx(0.2)
    assert split["hiring"] == pytest.approx(0.8)
    assert sum(split.values()) == pytest.approx(1.0)


def test_s5_score_view_includes_other_and_sums_to_one():
    r = record("s5", "other", "brand")
    answers = {
        "s5_score": {"0": 0.1, "1": 0.1, "2": 0.1, "3": 0.1, "4": 0.6},
        "s5_choice": {"browsing": 0.1, "experience": 0.1, "confirmation": 0.1, "own_and_service": 0.1, "other": 0.6},
    }
    views = final_distributions(SPECS["s5"], "both", r, answers)
    assert views["score"]["other"] == pytest.approx(0.6)
    assert views["score"]["initial_exploration"] == 0.0
    assert sum(views["score"].values()) == pytest.approx(1.0)
    assert views["choice"] == pytest.approx(views["score"])


def test_noul_view_maps_true_false_to_labels():
    views = final_distributions(SPECS["s4"], "noul", record("s4", "brand"), {"s4_noul": {"true": 0.9, "false": 0.1}})
    assert views["noul"] == {"brand": pytest.approx(0.9), "nonbrand": pytest.approx(0.1)}


# ----- metrics and calibration -------------------------------------------------

def test_summarize_perfect_and_auto_rate():
    labels = ["a", "b"]
    dists = [{"a": 0.9, "b": 0.1}, {"a": 0.2, "b": 0.8}, {"a": 0.6, "b": 0.4}]
    result = summarize(dists, labels, ["a", "b", "b"])
    assert result["accuracy"] == pytest.approx(2 / 3)
    probs = np.array([[0.99, 0.01], [0.9, 0.1], [0.6, 0.4]])
    assert auto_rate(probs, np.array([0, 0, 1])) == pytest.approx(2 / 3)


def test_paired_bootstrap_sign():
    result = paired_bootstrap([1, 1, 1, 0] * 50, [1, 0, 0, 0] * 50, samples=500)
    assert result["diff"] == pytest.approx(50.0)
    assert result["low"] > 0


def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(0)
    true_t = 2.5
    items = []
    for _ in range(3000):
        logits = rng.normal(0, 3, size=4)
        gold = rng.choice(4, p=apply_temperature(logits, true_t))
        items.append((logits.tolist(), int(gold)))
    assert fit_temperature(items) == pytest.approx(true_t, rel=0.1)


# ----- run matrix --------------------------------------------------------------

def test_stage1_matrix_matches_design_doc():
    runs = expand_runs("clef-flash", GROUPS, TASKS, stages={1})
    assert len(runs) == 27
    assert {r.group for r in runs} >= {"S-s4-noul", "S-s2-split", "S-s5-both"}
    with_optional = expand_runs("clef-flash", GROUPS, TASKS, stages={1}, include_optional=True)
    assert len(with_optional) == 30


def test_baseline_runs_are_eval_only():
    runs = expand_runs("kev-9b", GROUPS, TASKS, stages={0})
    assert all(not r.train and r.seed == 0 for r in runs)
    assert len(runs) == 3 + 4 + 3


def test_stage2_needs_selected_levels_then_expands():
    with pytest.raises(ValueError, match="selected_levels"):
        expand_runs("kev-9b", GROUPS, TASKS, stages={2})
    groups = copy.deepcopy(GROUPS)
    groups["selected_levels"] = {"s2": "choice", "s4": "noul", "s5": "score"}
    runs = expand_runs("kev-9b", groups, TASKS, stages={2})
    assert len(runs) == 12
    ko = [r for r in runs if r.group == "J+R-ko"][0]
    assert ko.train_langs == ("ko",) and ko.eval_langs == ("ko", "ja", "en")
    assert ko.replay == 2000 and ko.levels == {"s2": "choice", "s4": "noul", "s5": "score"}


def test_s5_score_text_matches_choice():
    """S5 choice and score differ only in question type: same instructions, same option text and order."""
    for variant in ("brand", "nonbrand"):
        choice = SPECS["s5"].question("s5_choice", variant)
        score = SPECS["s5"].question("s5_score", variant)
        assert score["instructions"] == choice["instructions"]
        assert score["levels"] == list(choice["criteria"])
        assert score["criteria"] == [f"{label}: {text}" for label, text in choice["criteria"].items()]
