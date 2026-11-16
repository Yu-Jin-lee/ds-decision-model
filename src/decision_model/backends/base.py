"""Backend interface. A backend knows how to fine-tune one family of decision
models and how to get raw per-option scores out of it. Everything else
(data, question formats, calibration, metrics) is shared."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# One prediction per record: {"id": ..., "answers": {question_id: {option_key: log_prob}}}
Prediction = dict[str, Any]


class Backend(ABC):
    def __init__(self, model_config: dict[str, Any]):
        self.config = model_config
        self.train_config = dict(model_config.get("train") or {})

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
