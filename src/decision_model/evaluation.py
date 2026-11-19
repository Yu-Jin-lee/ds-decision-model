"""Turn raw predictions into calibrated final-label distributions and metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .backends.base import Prediction, label_key
from .calibration import apply_temperature, fit_temperature
from .formats import INVALID, TaskSpec, final_distributions
from .metrics import stage_error, summarize


def calibration_items(records: list[dict[str, Any]], predictions: list[Prediction]) -> list[tuple[list[float], int]]:
    """(scores, gold index) for every labelled question, for temperature fitting."""
    by_id = {p["id"]: p for p in predictions}
    items = []
    for record in records:
        answers = by_id[record["id"]]["answers"]
        for question_id, question in record["questions"].items():
            if "label" not in question:
                continue
            options = answers[question_id]
            keys = list(options)
            items.append(([options[k] for k in keys], keys.index(label_key(question))))
    return items


def fit_model_temperature(records: list[dict[str, Any]], predictions: list[Prediction]) -> float:
    return fit_temperature(calibration_items(records, predictions))


def calibrated_answers(prediction: Prediction, temperature: float) -> dict[str, dict[str, float]]:
    answers = {}
    for question_id, options in prediction["answers"].items():
        keys = list(options)
        probabilities = apply_temperature([options[k] for k in keys], temperature)
        answers[question_id] = dict(zip(keys, probabilities.tolist()))
    return answers


def generated_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Metrics for a generative model's parsed answers: one label per record, no probabilities.

    An unparseable answer counts as wrong. Brier, ECE and the auto-processing rate need
    probabilities, so they are reported as None rather than computed from a one-hot guess.
    """
    n = len(rows)
    correct = sum(r["prediction"] == r["gold"] for r in rows)
    invalid = sum(r["prediction"] == INVALID for r in rows)
    return {"n": n, "accuracy": correct / n if n else float("nan"), "invalid_rate": invalid / n if n else float("nan"),
            "brier": None, "ece": None, "auto_rate_5pct": None}


def score(
    specs: dict[str, TaskSpec],
    records: list[dict[str, Any]],
    predictions: list[Prediction],
    temperature: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Per-record rows and per-cell metrics keyed ``task/lang/view``."""
    by_id = {p["id"]: p for p in predictions}
    rows: list[dict[str, Any]] = []
    for record in records:
        spec = specs[record["task"]]
        answers = calibrated_answers(by_id[record["id"]], temperature)
        views = final_distributions(spec, record["level"], record, answers)
        for view_name, distribution in views.items():
            rows.append(
                {
                    "id": record["id"],
                    "task": record["task"],
                    "lang": record["lang"],
                    "level": record["level"],
                    "view": view_name,
                    "variant": record.get("variant"),
                    "gold": record["gold"],
                    "prediction": max(distribution, key=distribution.get),
                    "distribution": distribution,
                }
            )

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["lang"], row["view"])].append(row)

    metrics: dict[str, Any] = {}
    for (task, lang, view), cell_rows in sorted(grouped.items()):
        labels = specs[task].final_labels
        if specs[task].view(view)["kind"] == "generated":
            cell = generated_summary(cell_rows)
        else:
            cell = summarize([r["distribution"] for r in cell_rows], labels, [r["gold"] for r in cell_rows])
        orders = specs[task].raw.get("stage_order")
        if orders:
            cell["stage_error"] = stage_error(
                [r["distribution"] for r in cell_rows],
                [orders[r["variant"]] for r in cell_rows],
                [r["gold"] for r in cell_rows],
            )
        metrics[f"{task}/{lang}/{view}"] = cell

    # Agreement between views of the same record (levels that train several formats together).
    agreement: dict[tuple[str, str], list[bool]] = defaultdict(list)
    per_record: dict[str, list[str]] = defaultdict(list)
    keys: dict[str, tuple[str, str]] = {}
    for row in rows:
        per_record[row["id"]].append(row["prediction"])
        keys[row["id"]] = (row["task"], row["lang"])
    for record_id, predictions_of_record in per_record.items():
        if len(predictions_of_record) > 1:
            agreement[keys[record_id]].append(len(set(predictions_of_record)) == 1)
    for (task, lang), values in agreement.items():
        metrics[f"{task}/{lang}/agreement"] = {"n": len(values), "agreement": sum(values) / len(values)}
    return rows, metrics
