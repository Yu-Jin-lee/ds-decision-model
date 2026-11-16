"""Clef backend on CPU with a stand-in for Cloudflare's joint_schema_model.

The fake mirrors the parts of the release module the backend relies on:
``encode_record`` (choice options sorted, noul true/false, score "0".."n"),
``collate_records``, and a model with ``language_model`` + ``head`` whose
forward returns per-question logits. It checks that labels map to the right
option, that training lowers the loss, and that a saved adapter and head load
back for prediction.
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from decision_model.backends.clef import ClefBackend, gold_indices, model_input  # noqa: E402
from decision_model.config import CONFIG_DIR, load_yaml  # noqa: E402
from decision_model.data import to_model_record  # noqa: E402
from decision_model.evaluation import fit_model_temperature, score  # noqa: E402
from decision_model.formats import load_task_specs  # noqa: E402

DIM = 16


@dataclass(frozen=True)
class Q:
    question_id: str
    option_ids: tuple


@dataclass(frozen=True)
class Enc:
    features: tuple
    questions: tuple


def _vector(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode()).digest()
    return [(b / 255.0) - 0.5 for b in digest[:DIM]]


def _options(question):
    if question["type"] == "noul":
        return ("true", "false")
    if question["type"] == "choice":
        return tuple(sorted(question["criteria"]))
    return tuple(str(i) for i in range(len(question["criteria"])))


def make_module() -> ModuleType:
    module = ModuleType("fake_joint_schema_model")

    def encode_record(tokenizer, record, max_length=16384, max_state_tokens=None):
        return Enc(
            features=tuple(_vector(json.dumps(record["state"], sort_keys=True))),
            questions=tuple(Q(qid, _options(q)) for qid, q in record["questions"].items()),
        )

    def collate_records(records, pad_token_id, device):
        return {"features": torch.tensor([r.features for r in records], device=device), "records": records}

    module.encode_record = encode_record
    module.collate_records = collate_records
    return module


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Sequential(torch.nn.Linear(DIM, DIM), torch.nn.Tanh(), torch.nn.Linear(DIM, DIM))
        self.config = SimpleNamespace(model_type="fake", to_dict=lambda: {"model_type": "fake"}, get=lambda *a, **k: None)

    def gradient_checkpointing_enable(self, **kwargs):
        pass

    def enable_input_require_grads(self):
        pass

    def forward(self, features):
        return self.model.language_model(features)


class FakeClef(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = Backbone()
        self.head = torch.nn.Linear(DIM, DIM)

    def forward(self, batch):
        base = self.language_model.get_base_model() if hasattr(self.language_model, "get_base_model") else self.language_model
        hidden = self.head(base.model.language_model(batch["features"]))
        out = []
        for row, record in zip(hidden, batch["records"]):
            out.append([torch.stack([row @ torch.tensor(_vector(f"{q.question_id}:{o}")) for o in q.option_ids]) for q in record.questions])
        return out


class FakeClefBackend(ClefBackend):
    def __init__(self, config, release_dir: Path):
        super().__init__(config, device="cpu")
        self.release_dir = release_dir
        self.module = make_module()
        torch.manual_seed(1234)
        self.initial = FakeClef().state_dict()

    def _load_release(self):
        model = FakeClef()
        model.load_state_dict(self.initial)
        processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
        return self.module, model, processor, self.release_dir


SPECS = load_task_specs(load_yaml(CONFIG_DIR / "tasks.yaml"))


def s4_records(n, level="both"):
    records = []
    for i in range(n):
        label = "brand" if i % 2 == 0 else "nonbrand"
        canonical = {"id": f"r{i}", "task": "s4", "lang": "ko", "state": {"keyword": f"kw{i}", "flag": label},
                     "label": label, "variant": None}
        records.append(to_model_record(canonical, SPECS["s4"], level, with_labels=True))
    return records


def test_gold_index_follows_sorted_choice_options():
    module = make_module()
    r = to_model_record({"id": "a", "task": "s2", "lang": "ko", "state": {}, "label": "person", "variant": None},
                        SPECS["s2"], "choice", with_labels=True)
    encoded = module.encode_record(None, model_input(r, with_labels=True))
    option_ids = encoded.questions[0].option_ids
    assert list(option_ids) == sorted(option_ids)
    assert option_ids[gold_indices(encoded, r)[0]] == "person"


def test_train_lowers_loss_and_checkpoint_round_trips(tmp_path):
    release = tmp_path / "release"
    release.mkdir()
    (release / "joint_head_config.json").write_text("{}")
    config = {"backend": "clef", "checkpoint": "fake/clef", "train": {
        "lora_rank": 4, "lora_alpha": 8, "lr": 5e-3, "head_lr": 5e-3, "epochs": 30, "batch": 4, "accum": 1,
        "backbone_dtype": "fp32", "head_dtype": "fp32", "gradient_checkpointing": True}}
    backend = FakeClefBackend(config, release)
    records = s4_records(16)

    baseline = backend.predict(None, records)
    checkpoint = backend.train(records, tmp_path / "ckpt", seed=0)
    meta = json.loads((checkpoint / "clef_checkpoint.json").read_text())
    assert meta["optimizer_steps"] == 30 * 4
    assert (checkpoint / "adapter" / "adapter_config.json").exists()
    assert (checkpoint / "joint_head.safetensors").exists()

    tuned = backend.predict(checkpoint, records)
    temperature = fit_model_temperature(records, tuned)
    _, before = score(SPECS, records, baseline, 1.0)
    _, after = score(SPECS, records, tuned, temperature)
    assert after["s4/ko/choice"]["accuracy"] > before["s4/ko/choice"]["accuracy"]
    assert after["s4/ko/choice"]["accuracy"] >= 0.9
    assert "s4/ko/agreement" in after


def test_pipeline_runs_one_cell_end_to_end(tmp_path):
    from decision_model.data import write_jsonl
    from decision_model.pipeline import execute, find_run

    data = tmp_path / "data"
    for lang in ("ko", "ja", "en"):
        for partition, n in (("train", 12), ("calibration", 6), ("development", 6), ("test", 6)):
            rows = []
            for i in range(n):
                label = "brand" if i % 2 == 0 else "nonbrand"
                rows.append({"id": f"{lang}:{partition}:{i}", "task": "s4", "lang": lang, "partition": partition,
                             "state": {"keyword": f"{lang}{i}", "flag": label}, "label": label, "variant": None})
            write_jsonl(data / "s4" / lang / f"{partition}.jsonl", rows)

    release = tmp_path / "release"
    release.mkdir()
    (release / "joint_head_config.json").write_text("{}")
    backend = FakeClefBackend({"backend": "clef", "checkpoint": "fake/clef", "train": {
        "lora_rank": 4, "lr": 5e-3, "head_lr": 5e-3, "epochs": 5, "batch": 4, "accum": 1,
        "backbone_dtype": "fp32", "head_dtype": "fp32"}}, release)

    run = find_run("clef-flash", "S-s4-both", 0)
    summary = execute(run, data, tmp_path / "runs", backend=backend)
    out = tmp_path / "runs" / "clef-flash" / "S-s4-both" / "s0"
    assert (out / "summary.json").exists() and (out / "scored_development.jsonl").exists()
    assert not (out / "scored_test.jsonl").exists()
    cells = summary["metrics_development"]
    assert {"s4/ko/noul", "s4/ja/choice", "s4/en/agreement"} <= set(cells)
    assert len(summary["data_manifest"]) == 12
