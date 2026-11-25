"""Chon training on the CPU with the chon package's tiny random Gemma 4 and a tiny release-style adapter package.

Checks that training reads every question exactly as chon.serve does (same token ids), that the loss falls, and that
the saved package loads in chon's DecisionModel the way chon.serve loads a release, with raw (temperature 1) reads.
Skipped when torch, peft or chon are not installed (pip install "chon @ git+https://github.com/xoai/chon").
"""

from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
chon = pytest.importorskip("chon")

from decision_model.backends import get_backend  # noqa: E402
from decision_model.backends.chon import gold_label, training_examples  # noqa: E402


def _tiny_module():
    """chon's tests/tiny_model.py (not installed with the package): found next to a source checkout."""
    import importlib.util

    for root in [Path(chon.__file__).resolve().parents[1], Path("/home/claude/chon-src")]:
        path = root / "tests" / "tiny_model.py"
        if path.exists():
            spec = importlib.util.spec_from_file_location("chon_tiny_model", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
    pytest.skip("chon's tests/tiny_model.py is not available (needs a source checkout of chon)")


def _records(n: int, seed: int = 0) -> list[dict]:
    """Our record format (state + labelled questions) over all three question types; every third record asks two
    questions about one state, so the request layout (header list) is exercised."""
    rng = random.Random(seed)
    records = []
    for i in range(n):
        weight, limit = rng.randint(1, 12), rng.randint(2, 9)
        labels = ["refund", "replace", "escalate", "close"]
        want = labels[weight % 4]
        questions = {
            "action": {"type": "choice", "instructions": "Which action does the ticket request?",
                       "criteria": {lab: f"the customer wants to {lab}" for lab in labels}, "label": want},
        }
        if i % 3 == 0:
            questions["limit"] = {"type": "noul", "instructions": "Is the parcel within the limit?",
                                  "criteria": {"true": "weight <= limit", "false": "weight > limit"},
                                  "label": weight <= limit}
        if i % 3 == 1:
            questions["severity"] = {"type": "score", "instructions": "How severe is it?",
                                     "criteria": ["low", "medium", "high"], "label": weight % 3}
        records.append({"id": f"r{i}", "state": {"ticket": i, "request": want, "weight": weight, "limit": limit},
                        "questions": questions})
    return records


@pytest.fixture(scope="module")
def package(tmp_path_factory) -> Path:
    """A release-style package: a LoRA adapter on the tiny base, its tokenizer, and a chon_config.json naming the base
    and carrying fitted settings that a fine-tuned package must drop."""
    from transformers import AutoModelForImageTextToText

    tiny = _tiny_module()
    base = tiny.build_tiny_model(str(tmp_path_factory.mktemp("tiny-base")))
    model = AutoModelForImageTextToText.from_pretrained(base, dtype=torch.float32)
    adapter = tiny.tiny_adapter(model)
    pkg = tmp_path_factory.mktemp("tiny-release")
    adapter.save_pretrained(pkg)
    for f in Path(base).iterdir():
        if f.name.startswith(("tokenizer", "special_tokens", "chat_template")):
            shutil.copy2(f, pkg / f.name)
    (pkg / "chon_config.json").write_text(json.dumps({
        "name": "tiny-chon", "base_model": base, "temperatures": {"choice": 0.5, "noul": 0.5, "score": 0.5},
        "post_temperatures": {"noul": 2.0}, "think": {"budget": 8}, "prefix_cache": True}))
    return pkg


def _backend(package: Path, **train):
    return get_backend({"name": "tiny-chon", "backend": "chon", "checkpoint": str(package), "base": "unused",
                        "dtype": "float32", "train": {"weights_dtype": "float32", "dtype": "fp32", "lr": 5e-3,
                                                      "epochs": 3, "batch": 4, "accum": 1, "checkpointing": 0, **train}})


def _release_dm(package: Path):
    from chon.infer import DecisionModel

    return DecisionModel.load(str(package), device="cpu", dtype="float32", think=False)


def test_gold_labels_use_chon_option_names():
    assert gold_label({"type": "noul", "label": True}) == "yes"
    assert gold_label({"type": "noul", "label": False}) == "no"
    assert gold_label({"type": "score", "label": 2}) == "2"
    assert gold_label({"type": "choice", "label": "close"}) == "close"


def test_training_reads_questions_exactly_as_serving(package, monkeypatch):
    """The token ids trained on are the ids DecisionModel.predict (what chon.serve calls) reads, question by question."""
    dm = _release_dm(package)
    records = _records(9)
    examples = training_examples(dm, records)
    served = []
    real = dm._logits

    def capture(seqs, n_options):
        served.extend(list(s) for s in seqs)
        return real(seqs, n_options)

    monkeypatch.setattr(dm, "_logits", capture)
    for record in records:
        questions = {n: {k: v for k, v in q.items() if k != "label"} for n, q in record["questions"].items()}
        dm.predict(record["state"], questions)
    assert [ids for ids, _, _ in examples] == served
    # the gold index points at the gold label in serving's option order
    first = records[0]["questions"]["action"]
    assert examples[0][2] == list(first["criteria"]).index(first["label"])


def test_train_lowers_loss_and_saves_a_servable_raw_package(package, tmp_path):
    from chon.infer import DecisionModel

    records = _records(24)
    backend = _backend(package)

    def mean_nll(dm):
        import math

        total, n = 0.0, 0
        for record in records:
            questions = {k: {kk: vv for kk, vv in q.items() if kk != "label"} for k, q in record["questions"].items()}
            dists = dm.distributions([(record["state"], q, None) for q in questions.values()], calibrated=False)
            for (name, q), dist in zip(record["questions"].items(), dists):
                total -= math.log(max(dist[gold_label(q)], 1e-12))
                n += 1
        return total / n

    before = mean_nll(_release_dm(package))
    checkpoint = backend.train(records, tmp_path / "checkpoint", seed=0)
    assert (checkpoint / "adapter_config.json").exists()
    assert (checkpoint / "tokenizer_config.json").exists()
    cfg = json.loads((checkpoint / "chon_config.json").read_text())
    assert cfg["base_model"] == json.loads((package / "chon_config.json").read_text())["base_model"]
    assert cfg["temperatures"] == {"choice": 1.0, "noul": 1.0, "score": 1.0}
    for dropped in ("post_temperatures", "think"):
        assert dropped not in cfg
    assert cfg["fine_tuned"]["questions"] == sum(len(r["questions"]) for r in records)

    # loaded the way chon.serve loads a release package: --model <folder>
    tuned = DecisionModel.load(str(checkpoint), device="cpu", dtype="float32")
    assert tuned.think is None and tuned.post_temperatures == {}
    after = mean_nll(tuned)
    assert after < before * 0.8, (before, after)


def test_serve_command_points_at_the_finetuned_package(tmp_path):
    backend = _backend(Path("pkg"))
    command = backend.serve_command(8000, tmp_path / "ckpt")
    assert command[command.index("--model") + 1] == str(tmp_path / "ckpt")
    assert backend.serve_command(8000)[backend.serve_command(8000).index("--model") + 1] == "pkg"


def test_replay_is_refused(package, tmp_path):
    with pytest.raises(NotImplementedError, match="replay"):
        _backend(package).train(_records(2), tmp_path, seed=0, replay=100)
