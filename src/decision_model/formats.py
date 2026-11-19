"""Question formats: build typed questions from a canonical record, and turn
model answers back into a distribution over the task's final labels.

A canonical record looks like::

    {"id": "kr:...", "task": "s4", "lang": "ko", "partition": "train",
     "state": {...}, "label": "brand", "variant": "brand"}

Every level of every task is described in ``configs/tasks.yaml``. This module
is the only place that knows how a format maps to final labels, so training,
evaluation and the stage-1 comparison all use the same rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

Distribution = dict[str, float]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    raw: dict[str, Any]

    @property
    def final_labels(self) -> list[str]:
        return list(self.raw["final_labels"])

    def levels(self) -> list[str]:
        return list(self.raw["levels"])

    def views_for(self, level: str) -> list[str]:
        try:
            return list(self.raw["levels"][level])
        except KeyError as error:
            raise KeyError(f"{self.task_id}: unknown level {level!r}") from error

    def view(self, name: str) -> dict[str, Any]:
        return self.raw["views"][name]

    def question(self, question_id: str, variant: str | None) -> dict[str, Any]:
        """Return the question spec with its variant (brand / non-brand) applied."""
        spec = dict(self.raw["questions"][question_id])
        variants = spec.pop("variants", None)
        if variants is not None:
            if variant is None:
                raise ValueError(f"{self.task_id}.{question_id} needs a record variant")
            spec.update(variants[variant])
        return spec


def load_task_specs(tasks_config: dict[str, Any]) -> dict[str, TaskSpec]:
    return {task_id: TaskSpec(task_id, raw) for task_id, raw in tasks_config["tasks"].items()}


def _labels_of(spec: dict[str, Any]) -> list[str]:
    """Final labels a single question can produce, in option order."""
    if spec["type"] == "noul":
        return [spec["true_label"], spec["false_label"]]
    if spec["type"] == "choice":
        return list(spec["criteria"])
    return list(spec["levels"])


def _question_payload(spec: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"type": spec["type"], "instructions": spec["instructions"]}
    if spec["type"] != "noul" or spec.get("criteria"):
        if spec.get("criteria") is not None:
            payload["criteria"] = spec["criteria"]
    return payload


def _answer_label(spec: dict[str, Any], final_label: str) -> Any:
    """Label in the question's own answer space, for a given final label."""
    if spec["type"] == "noul":
        if final_label == spec["true_label"]:
            return True
        if final_label == spec["false_label"]:
            return False
        raise ValueError(f"{final_label!r} is not an answer of noul question")
    if spec["type"] == "choice":
        if final_label not in spec["criteria"]:
            raise ValueError(f"{final_label!r} is not an option of choice question")
        return final_label
    return spec["levels"].index(final_label)


def build_questions(
    task: TaskSpec,
    level: str,
    record: dict[str, Any],
    with_labels: bool = True,
) -> dict[str, dict[str, Any]]:
    """Typed questions (Kev/SystemOne shape) for one record at one level."""
    variant = record.get("variant")
    final_label = record.get("label")
    questions: dict[str, dict[str, Any]] = {}
    for view_name in task.views_for(level):
        view = task.view(view_name)
        if view["kind"] == "single":
            question_id = view["question"]
            spec = task.question(question_id, variant)
            payload = _question_payload(spec)
            if with_labels:
                payload["label"] = _answer_label(spec, final_label)
            questions[question_id] = payload
        elif view["kind"] == "generated":
            # A generative model answered with its own production prompt: no question
            # type, options or wording from this file. The label is the final label.
            payload = {"type": "generated"}
            if with_labels:
                payload["label"] = final_label
            questions[view["question"]] = payload
        elif view["kind"] == "gate":
            gate_id, inner_id = view["gate"], view["inner"]
            gate_spec = task.question(gate_id, variant)
            gate_payload = _question_payload(gate_spec)
            is_gate = final_label == view["gate_label"]
            if with_labels:
                gate_payload["label"] = is_gate
            questions[gate_id] = gate_payload
            # The inner question is trained only on records the gate does not decide,
            # but it is always asked at prediction time.
            if not with_labels or not is_gate:
                inner_spec = task.question(inner_id, variant)
                inner_payload = _question_payload(inner_spec)
                if with_labels:
                    inner_payload["label"] = _answer_label(inner_spec, final_label)
                questions[inner_id] = inner_payload
        else:
            raise ValueError(f"unknown view kind {view['kind']!r}")
    return questions


def question_option_keys(spec: dict[str, Any]) -> list[str]:
    """Option keys as the model reports them (noul: true/false, score: "0".."n")."""
    if spec["type"] == "noul":
        return ["true", "false"]
    if spec["type"] == "choice":
        return list(spec["criteria"])
    return [str(index) for index in range(len(spec["criteria"]))]


def _single_distribution(spec: dict[str, Any], probabilities: Distribution) -> Distribution:
    keys = question_option_keys(spec)
    labels = _labels_of(spec)
    return {label: float(probabilities[key]) for key, label in zip(keys, labels)}


INVALID = "__invalid__"   # parsed label of a generated answer that matched no option


def final_distributions(
    task: TaskSpec,
    level: str,
    record: dict[str, Any],
    answers: dict[str, Distribution],
) -> dict[str, Distribution]:
    """One final-label distribution per view.

    ``answers`` maps question id to per-option probabilities keyed like
    :func:`question_option_keys` (noul: ``true``/``false``).
    Labels a view cannot produce get probability 0.
    """
    variant = record.get("variant")
    results: dict[str, Distribution] = {}
    for view_name in task.views_for(level):
        view = task.view(view_name)
        distribution = {label: 0.0 for label in task.final_labels}
        if view["kind"] == "single":
            spec = task.question(view["question"], variant)
            distribution.update(_single_distribution(spec, answers[view["question"]]))
        elif view["kind"] == "generated":
            # Keyed by final label; an unparseable output carries INVALID instead.
            distribution.update(answers[view["question"]])
        else:
            gate_probability = float(answers[view["gate"]]["true"])
            inner_spec = task.question(view["inner"], variant)
            inner = _single_distribution(inner_spec, answers[view["inner"]])
            for label, probability in inner.items():
                distribution[label] += (1.0 - gate_probability) * probability
            distribution[view["gate_label"]] += gate_probability
        results[view_name] = distribution
    return results
