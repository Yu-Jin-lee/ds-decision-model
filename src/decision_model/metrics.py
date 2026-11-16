"""Metrics on final-label distributions, and paired bootstrap comparisons."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _arrays(distributions: Sequence[dict[str, float]], labels: Sequence[str], gold: Sequence[str]):
    probabilities = np.array([[d[label] for label in labels] for d in distributions], dtype=float)
    totals = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / np.where(totals > 0, totals, 1.0)
    gold_index = np.array([list(labels).index(g) for g in gold])
    return probabilities, gold_index


def accuracy(probabilities: np.ndarray, gold_index: np.ndarray) -> float:
    return float((probabilities.argmax(axis=1) == gold_index).mean())


def brier(probabilities: np.ndarray, gold_index: np.ndarray) -> float:
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(gold_index)), gold_index] = 1.0
    return float(((probabilities - one_hot) ** 2).sum(axis=1).mean())


def ece(probabilities: np.ndarray, gold_index: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == gold_index
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(total)


def auto_rate(probabilities: np.ndarray, gold_index: np.ndarray, error_budget: float = 0.05) -> float:
    """Largest share of decisions, taken in order of confidence, whose error rate
    stays within the budget."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == gold_index
    order = np.argsort(-confidence, kind="stable")
    errors = np.cumsum(~correct[order])
    counts = np.arange(1, len(order) + 1)
    within = np.nonzero(errors / counts <= error_budget)[0]
    return float((within.max() + 1) / len(order)) if len(within) else 0.0


def summarize(
    distributions: Sequence[dict[str, float]],
    labels: Sequence[str],
    gold: Sequence[str],
) -> dict[str, float]:
    probabilities, gold_index = _arrays(distributions, labels, gold)
    return {
        "n": int(len(gold_index)),
        "accuracy": accuracy(probabilities, gold_index),
        "brier": brier(probabilities, gold_index),
        "ece": ece(probabilities, gold_index),
        "auto_rate_5pct": auto_rate(probabilities, gold_index),
    }


def stage_error(
    distributions: Sequence[dict[str, float]],
    orders: Sequence[Sequence[str]],
    gold: Sequence[str],
) -> float:
    """Mean distance in stages between the predicted and gold label, for records
    whose gold and predicted labels are both on that record's ordered scale (S5).
    ``orders`` gives the scale of each record (brand and non-brand differ)."""
    distances = []
    for distribution, order, gold_label in zip(distributions, orders, gold):
        predicted = max(distribution, key=distribution.get)
        if predicted in order and gold_label in order:
            distances.append(abs(order.index(predicted) - order.index(gold_label)))
    return float(np.mean(distances)) if distances else float("nan")


def paired_bootstrap(
    candidate_correct: Iterable[bool],
    reference_correct: Iterable[bool],
    samples: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Accuracy difference (candidate - reference, in points) with a 95% interval,
    resampling the same items for both models."""
    candidate = np.asarray(list(candidate_correct), dtype=float)
    reference = np.asarray(list(reference_correct), dtype=float)
    if candidate.shape != reference.shape:
        raise ValueError("candidate and reference must be scored on the same items")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(candidate), size=(samples, len(candidate)))
    differences = (candidate[index].mean(axis=1) - reference[index].mean(axis=1)) * 100
    return {
        "diff": float((candidate.mean() - reference.mean()) * 100),
        "low": float(np.percentile(differences, 2.5)),
        "high": float(np.percentile(differences, 97.5)),
    }
