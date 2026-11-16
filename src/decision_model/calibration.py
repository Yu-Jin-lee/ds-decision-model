"""Temperature scaling shared by every backend.

Backends report raw per-question log-probabilities (or logits). One temperature
per model is fitted on the calibration partition by minimising the negative
log-likelihood of the gold option, then applied to every question at prediction
time. Fitting on log-probabilities is equivalent to fitting on logits, because
softmax(log softmax(z) / T) == softmax(z / T).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def apply_temperature(scores: Sequence[float], temperature: float) -> np.ndarray:
    return _softmax(np.asarray(scores, dtype=float) / temperature)


def negative_log_likelihood(items: Sequence[tuple[Sequence[float], int]], temperature: float) -> float:
    total = 0.0
    for scores, gold in items:
        probabilities = apply_temperature(scores, temperature)
        total -= np.log(max(probabilities[gold], 1e-12))
    return total / max(len(items), 1)


def fit_temperature(
    items: Sequence[tuple[Sequence[float], int]],
    low: float = 0.05,
    high: float = 20.0,
) -> float:
    """Golden-section search over log T. ``items`` are (scores, gold option index)."""
    if not items:
        return 1.0
    ratio = (np.sqrt(5) - 1) / 2
    a, b = np.log(low), np.log(high)
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc = negative_log_likelihood(items, float(np.exp(c)))
    fd = negative_log_likelihood(items, float(np.exp(d)))
    for _ in range(60):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = negative_log_likelihood(items, float(np.exp(c)))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = negative_log_likelihood(items, float(np.exp(d)))
    return float(np.exp((a + b) / 2))
