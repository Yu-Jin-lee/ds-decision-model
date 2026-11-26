"""Laya backend (convaiinnovations/laya-multilingual checkpoint).

Laya ships its own training loop (``laya.train``); this backend calls it rather
than re-implementing it:

* ``train_model`` with a ``TrainConfig`` built from the model config. The
  defaults are the published recipe: the whole encoder and the decision head are
  trained, with the RLCD objective. ``freeze_encoder: true`` trains the head
  only, and ``loss: soft-ce`` trains on cross-entropy alone.
* Laya's own calibration split and temperature fit are not used. Calibration
  runs in shared code on the calibration partition, as for every backend.

Laya encodes each question in its own sequence with the state, so questions do
not see each other (like Kev, unlike Clef).
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .base import Backend, Prediction, safe_log, strip_labels

META_FILE = "laya_checkpoint.json"

# TrainConfig fields this backend passes through from the model config.
TRAIN_FIELDS = (
    "epochs", "micro_batch", "grad_accum", "encoder_lr", "head_lr", "min_lr", "weight_decay",
    "grad_clip", "loss", "rl_samples", "sigma_start", "sigma_end", "w_sph", "w_rps",
    "shuffle_options", "freeze_encoder", "amp", "gradient_checkpointing", "label_smoothing", "log_every",
)


def laya_option_keys(internal_question: dict[str, Any]) -> list[str]:
    """Option keys, in the order Laya's head scores them, as shared code expects them.

    Laya orders noul options [false, true], choice options as the criteria are
    listed, and score levels by index.
    """
    kind, criteria = internal_question["t"], internal_question["crit"]
    if kind == "noul":
        return ["false", "true"]
    if kind == "choice":
        return [str(key) for key in criteria]
    return [str(index) for index in range(len(criteria))]


def to_laya_row(record: dict[str, Any]) -> dict[str, Any]:
    """A labelled record as Laya's ``{state, questions, expected}`` training row."""
    questions = record["questions"]
    return {
        "state": record["state"],
        "questions": strip_labels(questions),
        "expected": {qid: q["label"] for qid, q in questions.items() if "label" in q},
    }


class LayaBackend(Backend):
    def __init__(self, model_config: dict[str, Any], device: str = "auto"):
        super().__init__(model_config)
        self.device = device

    def _lengths(self, cfg: dict[str, Any]) -> tuple[int, int]:
        max_len = int(self.train_config.get("max_len") or cfg.get("max_len", 512))
        head_max_len = int(self.train_config.get("head_max_len") or cfg.get("head_max_len", 192))
        return max_len, head_max_len

    def train(
        self,
        records: list[dict[str, Any]],
        out_dir: Path,
        seed: int,
        replay: int = 0,
        replay_records: list[dict[str, Any]] | None = None,
    ) -> Path:
        import random

        from laya.common import uses_parallel_layout
        from laya.train import TrainConfig, items_from_rows, load_checkpoint, resolve_device, save_checkpoint, train_model

        model, tokenizer, cfg = load_checkpoint(self.config["checkpoint"])
        device = resolve_device(self.device)
        model.to(device)
        max_len, head_max_len = self._lengths(cfg)

        rows = [to_laya_row(record) for record in records]
        if replay and replay_records:
            rng = random.Random(seed)
            rows += [to_laya_row(r) for r in rng.sample(replay_records, min(replay, len(replay_records)))]
        items, skipped = items_from_rows(tokenizer, rows, max_len, head_max_len)
        if not items:
            raise ValueError(f"no trainable Laya items (skipped: {skipped})")

        options = {k: self.train_config[k] for k in TRAIN_FIELDS if k in self.train_config}
        if "shuffle_options" in options:
            options["shuffle_options"] = tuple(options["shuffle_options"])
        config = TrainConfig(seed=seed, max_len=max_len, head_max_len=head_max_len, **options)
        history = train_model(
            model, tokenizer, items, config, device, max_len, head_max_len, parallel=uses_parallel_layout(cfg)
        )

        out_dir = Path(out_dir)
        save_checkpoint(model, tokenizer, dict(cfg, max_len=max_len, head_max_len=head_max_len), str(out_dir))
        meta = {
            "checkpoint": self.config["checkpoint"],
            "train": self.train_config,
            "seed": seed,
            "items": len(items),
            "skipped": skipped,
            "epoch_loss": history,
        }
        (out_dir / META_FILE).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        return out_dir

    def _load(self, checkpoint: Path | None) -> tuple[Any, Any, Any, Any]:
        from laya.train import load_checkpoint, resolve_device

        model, tokenizer, cfg = load_checkpoint(str(checkpoint) if checkpoint is not None else self.config["checkpoint"])
        device = resolve_device(self.device)
        model.to(device).eval()
        return model, tokenizer, cfg, device

    def predict(self, checkpoint: Path | None, records: list[dict[str, Any]]) -> list[Prediction]:
        loaded = self._load(checkpoint)
        return self._predict_loaded(loaded, records, int(self.train_config.get("eval_batch", 32)))

    def inference_precision(self) -> str:
        autocast = self.serving_config.get("autocast")
        return f"{autocast} autocast (serving)" if autocast else "fp32 (amp off)"

    def _autocast(self, device: Any) -> Any:
        """bf16 autocast for a serving run (``serving.autocast``); nothing for the fp32 evaluation path."""
        import contextlib

        import torch

        name = self.serving_config.get("autocast")
        if not name:
            return contextlib.nullcontext()
        dtype = {"bf16": torch.bfloat16, "bfloat16": torch.bfloat16, "fp16": torch.float16}[str(name)]
        return torch.autocast(device.type, dtype=dtype)

    @contextmanager
    def predictor(self, checkpoint: Path | None) -> Iterator[Callable[[dict[str, Any]], Prediction]]:
        loaded = self._load(checkpoint)
        # One record per call; all of its questions go through in one forward pass.
        yield lambda record: self._predict_loaded(loaded, [record], max(1, len(record["questions"])))[0]

    @contextmanager
    def batch_predictor(self, checkpoint: Path | None) -> Iterator[Callable[[list[dict[str, Any]]], list[Prediction]]]:
        loaded = self._load(checkpoint)
        # Several records per call; all of their questions go through in one forward pass.
        yield lambda records: self._predict_loaded(loaded, records, max(1, sum(len(r["questions"]) for r in records)))

    def _predict_loaded(self, loaded: tuple[Any, Any, Any, Any], records: list[dict[str, Any]], batch_size: int) -> list[Prediction]:
        import torch

        from laya.common import collate_items, uses_parallel_layout
        from laya.train import _forward, encode_item, encode_state, make_item, to_internal

        model, tokenizer, cfg, device = loaded
        max_len, head_max_len = self._lengths(cfg)
        parallel = uses_parallel_layout(cfg)

        # One item per (record, question). Questions Laya cannot encode (options that
        # collapse or do not fit the head budget) get a uniform answer and are counted.
        pending: list[tuple[int, str, list[str], dict[str, Any]]] = []
        predictions: list[Prediction] = []
        for index, record in enumerate(records):
            state_ids = encode_state(tokenizer, record["state"], max_len)
            prediction: Prediction = {"id": record["id"], "answers": {}}
            for qid, question in strip_labels(record["questions"]).items():
                internal = to_internal(qid, question)
                keys = laya_option_keys(internal)
                uniform = [1.0 / len(keys)] * len(keys)
                item, reason = make_item(tokenizer, internal, uniform, state_ids, head_max_len, max_len)
                if item is None:
                    prediction["answers"][qid] = {k: safe_log(1.0 / len(keys)) for k in keys}
                    prediction.setdefault("skipped", {})[qid] = reason
                else:
                    pending.append((index, qid, keys, item))
            predictions.append(prediction)

        with torch.no_grad():
            for start in range(0, len(pending), batch_size):
                chunk = pending[start : start + batch_size]
                encoded = [encode_item(tokenizer, item, max_len, head_max_len, parallel=parallel) for *_, item in chunk]
                batch = collate_items([encoded], tokenizer.pad_token_id)
                with self._autocast(device):
                    logits = _forward(model, batch, device, amp=False, detach_encoder=False).cpu()
                for (index, qid, keys, _), row in zip(chunk, logits):
                    log_probs = torch.log_softmax(row[: len(keys)], -1).tolist()
                    predictions[index]["answers"][qid] = dict(zip(keys, log_probs))
        return predictions

