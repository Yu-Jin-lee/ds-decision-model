"""Backend interface. A backend knows how to fine-tune one family of decision
models and how to get raw per-option scores out of it. Everything else
(data, question formats, calibration, metrics) is shared."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

# One prediction per record: {"id": ..., "answers": {question_id: {option_key: log_prob}}}
Prediction = dict[str, Any]


class Backend(ABC):
    # True when ``predictor``'s function only forwards the record to a server the backend started
    # (kev.serve, chon.serve, vLLM): several threads may call it at once, and that server decides how to
    # batch them. False for a model run in this process, which the endpoint calls one batch at a time.
    concurrent_predictor = False

    def __init__(self, model_config: dict[str, Any]):
        self.config = model_config
        self.train_config = dict(model_config.get("train") or {})
        # Serving profile (configs/serving.yaml): predict as the model would be served, bf16 with its own
        # fastest engine, instead of the evaluation path. Set by get_backend(..., serving=True).
        self.serving = False

    @property
    def serving_config(self) -> dict[str, Any]:
        """The model config's ``serving`` section (empty when the run is not a serving run)."""
        return dict(self.config.get("serving") or {}) if self.serving else {}

    @abstractmethod
    def train(
        self,
        records: list[dict[str, Any]],
        out_dir: Path,
        seed: int,
        replay: int = 0,
        replay_records: list[dict[str, Any]] | None = None,
    ) -> Path:
        """Fine-tune from the released checkpoint and return the checkpoint directory."""

    @abstractmethod
    def predict(self, checkpoint: Path | None, records: list[dict[str, Any]]) -> list[Prediction]:
        """Raw (uncalibrated) log-probabilities per option for every question.

        ``checkpoint=None`` means the released model with no fine-tuning (B0).
        Option keys follow :func:`decision_model.formats.question_option_keys`.
        """

    def inference_precision(self) -> str:
        """Numeric precision the model runs at when predicting (recorded with latency)."""
        return "unknown"

    @contextmanager
    def predictor(self, checkpoint: Path | None) -> Iterator[Callable[[dict[str, Any]], Prediction]]:
        """Load the model once and yield a function that predicts one record.

        Used for latency measurement (decision_model.latency): the model is loaded
        before timing starts and each call handles exactly one record, all of its
        questions, with the backend's evaluation settings.
        """
        raise NotImplementedError(f"{type(self).__name__} has no predictor")
        yield  # pragma: no cover

    def has_batch_predictor(self) -> bool:
        return type(self).batch_predictor is not Backend.batch_predictor

    @contextmanager
    def batch_predictor(self, checkpoint: Path | None) -> Iterator[Callable[[list[dict[str, Any]]], list[Prediction]]]:
        """Like ``predictor``, but each call predicts several records at once (the endpoint's micro-batching,
        decision_model.endpoint). Only for a model run in this process."""
        raise NotImplementedError(f"{type(self).__name__} has no batch predictor")
        yield  # pragma: no cover


def label_key(question: dict[str, Any]) -> str:
    """Gold option key of a labelled question."""
    label = question["label"]
    if question["type"] == "noul":
        return "true" if label else "false"
    if question["type"] == "score":
        return str(int(label))
    return str(label)


def strip_labels(questions: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {qid: {k: v for k, v in q.items() if k != "label"} for qid, q in questions.items()}


def safe_log(probability: float) -> float:
    return math.log(min(max(probability, 1e-12), 1.0))
