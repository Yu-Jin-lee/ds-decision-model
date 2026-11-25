"""Inference latency under the same conditions for every model.

What is held fixed, so the numbers compare across models and runs:

* Records: a fixed sample of the run's development records, ``per_cell`` per
  (task, language) cell, picked by a hash of the record id. Every model sees the
  same records in the same order for the same group.
* Unit: one record end to end. The timer starts with the record (state and
  questions) and stops when every question's option probabilities are back, so
  tokenisation is included and formats with more questions take longer.
* Batch size 1, records one after another. Kev's serving API takes one record
  per request, so batch 1 is the only setting all three models share.
* The model is already loaded and ``warmup`` records are run first and not
  counted. They are records outside the timed sample (the next ones in hash
  order), so a server that caches whole prompts (vLLM's prefix cache) has never
  seen a timed record before it is timed. Model loading, calibration and scoring
  are not timed.
* GPU work is finished before the clock is read (``torch.cuda.synchronize``).
* Same instance type for every run (configs/models/*.yaml); the environment is
  recorded next to the numbers.
"""

from __future__ import annotations

import hashlib
import os
import platform
import statistics
import time
from collections import defaultdict
from typing import Any, Callable

PER_CELL = 30
WARMUP = 10


def sample_records(records: list[dict[str, Any]], per_cell: int = PER_CELL) -> list[dict[str, Any]]:
    """``per_cell`` records per (task, lang), ordered by a hash of the id."""
    cells: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        cells[(record["task"], record["lang"])].append(record)
    picked: list[dict[str, Any]] = []
    for key in sorted(cells):
        ordered = sorted(cells[key], key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        picked += ordered[:per_cell]
    return sorted(picked, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())


def _hash_order(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())


def warmup_records(records: list[dict[str, Any]], sample: list[dict[str, Any]], warmup: int = WARMUP) -> list[dict[str, Any]]:
    """``warmup`` records that are not in the timed sample, in hash order (falls back to the sample when the
    data has no others, e.g. a tiny test set)."""
    timed = {id(r) for r in sample}
    others = _hash_order([r for r in records if id(r) not in timed])
    return (others or _hash_order(sample))[:warmup]


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def _stats(ms: list[float]) -> dict[str, float]:
    return {
        "records": len(ms),
        "mean_ms": round(statistics.fmean(ms), 2),
        "p50_ms": round(_percentile(ms, 0.50), 2),
        "p95_ms": round(_percentile(ms, 0.95), 2),
        "records_per_s": round(1000.0 / statistics.fmean(ms), 2),
    }


def environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "instance_type": os.environ.get("SM_CURRENT_INSTANCE_TYPE") or os.environ.get("DECISION_MODEL_INSTANCE_TYPE"),
        "python": platform.python_version(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return env


def measure(
    predict_one: Callable[[dict[str, Any]], Any],
    records: list[dict[str, Any]],
    per_cell: int = PER_CELL,
    warmup: int = WARMUP,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Time ``predict_one`` on the fixed sample; ``predict_one`` takes one record."""
    sample = sample_records(records, per_cell)
    if not sample:
        return {"records": 0}
    warm = warmup_records(records, sample, warmup)
    for record in warm:
        predict_one(record)
    _sync()
    timed_ids = {id(r) for r in sample}

    by_task: dict[str, list[float]] = defaultdict(list)
    all_ms: list[float] = []
    questions = []
    for record in sample:
        start = clock()
        predict_one(record)
        _sync()
        elapsed = (clock() - start) * 1000.0
        all_ms.append(elapsed)
        by_task[record["task"]].append(elapsed)
        questions.append(len(record["questions"]))

    result = {
        **_stats(all_ms),
        "questions_per_record": round(statistics.fmean(questions), 2),
        "by_task": {task: _stats(ms) for task, ms in sorted(by_task.items())},
        "conditions": {
            "per_cell": per_cell,
            "warmup": len(warm),
            "warmup_outside_sample": all(id(r) not in timed_ids for r in warm),
            "batch_size": 1,
            "unit": "one record, all its questions, end to end",
            "sample_sha256": hashlib.sha256("\n".join(r["id"] for r in sample).encode()).hexdigest(),
        },
        "environment": environment(),
    }
    return result


def view_question_ids(view: dict[str, Any]) -> list[str]:
    """Question ids one view asks (single: its question; gate: gate + inner)."""
    if view["kind"] == "single":
        return [view["question"]]
    return [view["gate"], view["inner"]]


def single_view_records(
    records: list[dict[str, Any]], task: str, question_ids: list[str]
) -> list[dict[str, Any]]:
    """The task's records with only the given questions, as if production asked just those."""
    out = []
    for record in records:
        if record["task"] != task:
            continue
        questions = {qid: q for qid, q in record["questions"].items() if qid in question_ids}
        if questions:
            out.append({**record, "questions": questions})
    return out


def measure_by_view(
    predict_one: Callable[[dict[str, Any]], Any],
    records: list[dict[str, Any]],
    specs: dict[str, Any],
    levels: dict[str, str],
    **kwargs: Any,
) -> dict[str, Any]:
    """Latency when a level that trains several formats is asked one format at a time.

    A model trained on ``both`` may be served with a single question; this times
    exactly that, keyed ``task/view``. Levels with one view are skipped (the main
    measurement already covers them).
    """
    results: dict[str, Any] = {}
    for task, level in levels.items():
        view_names = specs[task].views_for(level)
        if len(view_names) < 2:
            continue
        for view_name in view_names:
            ids = view_question_ids(specs[task].view(view_name))
            subset = single_view_records(records, task, ids)
            if subset:
                results[f"{task}/{view_name}"] = measure(predict_one, subset, **kwargs)
    return results
