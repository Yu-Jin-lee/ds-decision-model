"""Laya backend without the laya package: a stand-in for the few laya functions
the backend calls, to check option order (noul is [false, true] in Laya),
label conversion, skipped questions and batching. The real library is
exercised in the smoke run."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from decision_model.backends import get_backend  # noqa: E402
from decision_model.backends.laya import LayaBackend, laya_option_keys, to_laya_row  # noqa: E402
from decision_model.config import CONFIG_DIR, expand_runs, load_model_config, load_yaml  # noqa: E402
from decision_model.data import to_model_record  # noqa: E402
from decision_model.formats import load_task_specs  # noqa: E402

SPECS = load_task_specs(load_yaml(CONFIG_DIR / "tasks.yaml"))


def test_option_keys_follow_laya_order():
    assert laya_option_keys({"t": "noul", "crit": None}) == ["false", "true"]
    assert laya_option_keys({"t": "choice", "crit": {"b": None, "a": "x"}}) == ["b", "a"]
    assert laya_option_keys({"t": "score", "crit": ["low", "mid", "high"]}) == ["0", "1", "2"]


def test_to_laya_row_moves_labels_to_expected():
    record = to_model_record({"id": "a", "task": "s4", "lang": "ko", "state": {"keyword": "k"},
                              "label": "brand", "variant": None}, SPECS["s4"], "both", with_labels=True)
    row = to_laya_row(record)
    assert row["expected"] == {"s4_noul": True, "s4_choice": "brand"}
    assert all("label" not in q for q in row["questions"].values())


def test_config_and_registry():
    config = load_model_config("laya")
    assert isinstance(get_backend(config), LayaBackend)
    assert len(expand_runs("laya", load_yaml(CONFIG_DIR / "groups.yaml"), load_yaml(CONFIG_DIR / "tasks.yaml"), stages={1})) == 27


@pytest.fixture
def fake_laya(monkeypatch):
    """Minimal laya.train / laya.common. Logit for option i is i (so the last option wins)."""
    common = ModuleType("laya.common")
    train = ModuleType("laya.train")
    common.uses_parallel_layout = lambda cfg: False
    common.collate_items = lambda batch, pad: {"k": [item["k"] for item in batch[0]]}

    def to_internal(qid, question):
        crit = question.get("criteria")
        if question["type"] == "choice" and isinstance(crit, list):
            crit = {c: None for c in crit}
        return {"t": question["type"], "ins": question["instructions"], "crit": crit}

    def make_item(tok, q, target, state_ids, head_max_len, max_len=None):
        if q["ins"] == "too long":
            return None, "options_beyond_max_len"
        return {"q": q, "k": len(target), "target": target}, None

    def forward(model, batch, device, amp, detach_encoder):
        width = max(batch["k"])
        return torch.tensor([[float(i) for i in range(width)] for _ in batch["k"]])

    train.to_internal = to_internal
    train.make_item = make_item
    train.encode_state = lambda tok, state, max_len: [1, 2, 3]
    train.encode_item = lambda tok, item, max_len, head_max_len, parallel=False: item
    train._forward = forward
    train.resolve_device = lambda device: torch.device("cpu")
    model = SimpleNamespace(to=lambda device: SimpleNamespace(eval=lambda: None))
    train.load_checkpoint = lambda path: (model, SimpleNamespace(pad_token_id=0), {"max_len": 1024, "head_max_len": 256})
    package = ModuleType("laya")
    monkeypatch.setitem(sys.modules, "laya", package)
    monkeypatch.setitem(sys.modules, "laya.common", common)
    monkeypatch.setitem(sys.modules, "laya.train", train)


def test_predict_maps_logits_to_keys_and_counts_skipped(fake_laya):
    backend = LayaBackend({"backend": "laya", "checkpoint": "multilingual", "train": {"eval_batch": 2}}, device="cpu")
    records = [
        to_model_record({"id": f"r{i}", "task": "s4", "lang": "ko", "state": {"keyword": "k"}, "label": "brand",
                         "variant": None}, SPECS["s4"], "both", with_labels=True)
        for i in range(3)
    ]
    records[2]["questions"]["s4_choice"]["instructions"] = "too long"
    predictions = backend.predict(None, records)
    noul = predictions[0]["answers"]["s4_noul"]
    assert set(noul) == {"false", "true"} and noul["true"] > noul["false"]   # true is Laya's 2nd slot
    choice = predictions[1]["answers"]["s4_choice"]
    assert list(choice) == ["brand", "nonbrand"] and choice["nonbrand"] > choice["brand"]
    assert predictions[2]["skipped"] == {"s4_choice": "options_beyond_max_len"}
    assert predictions[2]["answers"]["s4_choice"]["brand"] == pytest.approx(predictions[2]["answers"]["s4_choice"]["nonbrand"])
