"""Endpoint load test: which records go in which step, and what the answers say (no Locust here).

Two methods share everything here and differ only in where the model is served and how it is called:

* ``job``: a SageMaker training job serves the model on localhost and runs Locust beside it (bench/job/).
* ``endpoint``: a SageMaker real-time endpoint, called with InvokeEndpoint from another machine
  (bench/endpoint/).

What is held fixed, so the numbers compare across models (configs/bench.yaml):

* Records: the run's records from ``partitions`` (never test), labels removed, in the order of a hash of
  their id. Every model of the same task gets the same records in the same steps.
* Steps: closed loop with ``users`` clients and no think time. Step k sends its own slice of the pool, so a
  record is timed once (``records_reused`` says when the pool was too small). Warm-up records come from the
  end of the pool and are not counted.
* Unit: one record end to end through HTTP, all of its questions; ``ms`` is measured by the client.
* Steady state: the first and last ``users`` answers of a step (ramp up, drain) are left out. Throughput is
  the rate of answers between the first and the last answer that are kept.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .config import CONFIG_DIR, load_yaml
from .latency import _percentile

BENCH_DIR = "bench"   # results go to {out_root}/bench/{method}/{model}/{group}/s{seed}/
METHODS = ("job", "endpoint")
SENT_KEYS = ("id", "task", "lang", "level", "variant", "state", "questions")
STEP_COLUMNS = [
    "users", "requests", "ok", "failed", "error_rate", "steady_records", "records_per_s", "questions_per_s",
    "overall_records_per_s", "mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms", "server_mean_ms",
    "server_p95_ms", "mean_batch", "step_seconds", "client_cpu", "timed_out",
]


def load_bench_config(config_dir: Path = CONFIG_DIR) -> dict[str, Any]:
    return load_yaml(config_dir / "bench.yaml")


def result_dir(out_root: str | Path, method: str, run_id: str, label: str | None = None) -> Path:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    return Path(out_root) / BENCH_DIR / method / run_id / (label or "")


def build_plan(run: Any, data_root: str | Path, config: dict[str, Any], users: list[int] | None = None,
               requests_per_step: int | None = None, warmup: int | None = None) -> dict[str, Any]:
    """The run's records (its tasks, level and languages) from the configured partitions, planned into steps.
    Flags override configs/bench.yaml."""
    from .data import build_records
    from .formats import load_task_specs

    users = [int(u) for u in (users or config["users"])]
    requests_per_step = int(requests_per_step or config["requests_per_step"])
    warmup = int(config["warmup"] if warmup is None else warmup)
    partitions = check_partitions(config["partitions"])
    specs = load_task_specs(load_yaml(CONFIG_DIR / "tasks.yaml"))
    records = []
    for partition in partitions:
        records += build_records(data_root, specs, run.levels, run.eval_langs, partition,
                                 with_labels=False, for_prediction=True)
    plan = plan_steps(record_pool(records), users, requests_per_step, int(config["min_requests_per_user"]), warmup)
    plan["conditions"] = {
        "users": users,
        "requests_per_step": requests_per_step,
        "min_requests_per_user": int(config["min_requests_per_user"]),
        "warmup": len(plan["warmup"]),
        "partitions": partitions,
        "pool_size": plan["pool_size"],
        "records_reused": plan["records_reused"],
        "sample_sha256": plan["sample_sha256"],
        "max_step_seconds": config["max_step_seconds"],
        "unit": "one record, all its questions, end to end (client clock)",
        "loop": "closed: each user sends the next record when the answer arrives, no think time",
    }
    return plan


def describe_plan(run_id: str, plan: dict[str, Any]) -> str:
    return (f"{run_id}: {plan['pool_size']} records, {len(plan['warmup'])} warm-up, steps "
            f"{[(s['users'], len(s['records'])) for s in plan['steps']]}, reused {plan['records_reused']}")


def make_summary(method: str, run: Any, plan: dict[str, Any], steps: list[dict[str, Any]], endpoint: dict[str, Any],
                 config: dict[str, Any], label: str | None = None, checkpoint: str | None = None,
                 **extra: Any) -> dict[str, Any]:
    """summary.json of one load test; the same keys for both methods (bench/compare.py reads them)."""
    import os

    from .latency import environment
    from .pipeline import _git_commit, _git_dirty

    return {
        "method": method,
        "run": run.to_dict(),
        "model": run.model,
        "group": run.group,
        "seed": run.seed,
        "tasks": sorted(run.levels),
        "label": label,
        "checkpoint": checkpoint,
        "endpoint": endpoint,
        "conditions": plan["conditions"],
        "client_environment": {**environment(), "cpu_count": os.cpu_count()},
        "steps": steps,
        "best": best_steps(steps, config.get("slo_p95_ms")),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        **extra,
    }


def check_partitions(partitions: Iterable[str]) -> list[str]:
    partitions = list(partitions)
    if "test" in partitions:
        raise ValueError("the test partition is only scored once, for the final candidate (--final); "
                         "it is never used for the load test")
    return partitions


def request_body(record: dict[str, Any]) -> dict[str, Any]:
    """What is sent: the record without its gold label or question labels."""
    body = {key: record[key] for key in SENT_KEYS if key in record}
    body["questions"] = {qid: {k: v for k, v in q.items() if k != "label"} for qid, q in record["questions"].items()}
    return body


def record_pool(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Request bodies in the order of a hash of the record id (input order does not matter)."""
    return [request_body(r) for r in sorted(records, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())]


def step_size(users: int, requests_per_step: int, min_requests_per_user: int) -> int:
    return max(int(requests_per_step), int(min_requests_per_user) * int(users))


def plan_steps(pool: list[dict[str, Any]], users_levels: list[int], requests_per_step: int,
               min_requests_per_user: int, warmup: int) -> dict[str, Any]:
    """Warm-up records (end of the pool) and one slice of the pool per step (from the start)."""
    if not pool:
        raise ValueError("no records to send")
    warmup = min(int(warmup), max(0, len(pool) - 1))
    warm = pool[len(pool) - warmup:] if warmup else []
    timed = pool[: len(pool) - warmup]
    steps, cursor, reused = [], 0, 0
    for users in users_levels:
        size = step_size(users, requests_per_step, min_requests_per_user)
        items = [timed[(cursor + i) % len(timed)] for i in range(size)]
        reused += max(0, min(size, cursor + size - len(timed)))
        cursor += size
        steps.append({"users": int(users), "records": items})
    return {"warmup": warm, "steps": steps, "records_reused": reused, "pool_size": len(pool),
            "sample_sha256": hashlib.sha256("\n".join(r["id"] for r in timed[:cursor]).encode()).hexdigest()}


def _latency_stats(ms: list[float]) -> dict[str, float | None]:
    if not ms:
        return {k: None for k in ("mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms")}
    return {
        "mean_ms": round(statistics.fmean(ms), 2),
        "p50_ms": round(_percentile(ms, 0.50), 2),
        "p90_ms": round(_percentile(ms, 0.90), 2),
        "p95_ms": round(_percentile(ms, 0.95), 2),
        "p99_ms": round(_percentile(ms, 0.99), 2),
        "max_ms": round(max(ms), 2),
    }


def steady_rows(rows: list[dict[str, Any]], users: int) -> list[dict[str, Any]]:
    """Successful answers without the first and last ``users`` (ramp up, drain), in order of arrival.
    A step too short to drop them keeps all of its answers."""
    ok = sorted((r for r in rows if r["ok"]), key=lambda r: r["end_s"])
    if len(ok) > 2 * users + 1:
        return ok[users: len(ok) - users]
    return ok


def _rate(rows: list[dict[str, Any]]) -> float | None:
    """Answers per second between the first and the last answer in ``rows`` (sorted by end_s)."""
    if len(rows) < 2:
        return None
    span = rows[-1]["end_s"] - rows[0]["end_s"]
    return round((len(rows) - 1) / span, 2) if span > 0 else None


def summarize_step(rows: list[dict[str, Any]], users: int, step_seconds: float | None = None,
                   client_cpu: float | None = None, timed_out: bool = False) -> dict[str, Any]:
    ok = [r for r in rows if r["ok"]]
    steady = steady_rows(rows, users)
    rate = _rate(steady)
    questions = statistics.fmean(r["questions"] for r in steady) if steady else None
    server_ms = [r["server_ms"] for r in steady if r.get("server_ms") is not None]
    batches = [r["batch"] for r in steady if r.get("batch") is not None]
    span = (max(r["end_s"] for r in rows) - min(r["start_s"] for r in rows)) if rows else 0.0
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in steady:
        by_task[row["task"]].append(row["ms"])
    return {
        "users": users,
        "requests": len(rows),
        "ok": len(ok),
        "failed": len(rows) - len(ok),
        "error_rate": round((len(rows) - len(ok)) / len(rows), 4) if rows else None,
        "steady_records": len(steady),
        "records_per_s": rate,
        "questions_per_s": round(rate * questions, 2) if rate is not None and questions is not None else None,
        "overall_records_per_s": round(len(ok) / span, 2) if span > 0 else None,
        **_latency_stats([r["ms"] for r in steady]),
        "server_mean_ms": round(statistics.fmean(server_ms), 2) if server_ms else None,
        "server_p95_ms": round(_percentile(server_ms, 0.95), 2) if server_ms else None,
        "mean_batch": round(statistics.fmean(batches), 2) if batches else None,
        "questions_per_record": round(questions, 2) if questions is not None else None,
        "step_seconds": round(step_seconds, 2) if step_seconds is not None else None,
        "client_cpu": round(client_cpu, 3) if client_cpu is not None else None,
        "timed_out": timed_out,
        "by_task": {task: _latency_stats(ms) | {"records": len(ms)} for task, ms in sorted(by_task.items())},
    }


def best_steps(steps: list[dict[str, Any]], slo_p95_ms: float | None) -> dict[str, Any]:
    """The step with the most records/s, and the one with the most records/s whose p95 meets the target
    (no failed requests in either)."""
    clean = [s for s in steps if s.get("records_per_s") and not s.get("failed")]

    def pick(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        if not candidates:
            return None
        best = max(candidates, key=lambda s: s["records_per_s"])
        return {k: best[k] for k in ("users", "records_per_s", "questions_per_s", "p50_ms", "p95_ms")}

    within = [s for s in clean if slo_p95_ms is not None and s["p95_ms"] is not None and s["p95_ms"] <= slo_p95_ms]
    single = next((s for s in steps if s["users"] == 1), None)
    return {
        "max_throughput": pick(clean),
        "within_slo": pick(within),
        "slo_p95_ms": slo_p95_ms,
        "single_user": ({k: single[k] for k in ("p50_ms", "p95_ms", "records_per_s")} if single else None),
    }


def write_results(out_dir: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    """``requests.jsonl.gz`` (every request), ``steps.csv`` (one row per step), then ``summary.json``
    last, so its presence means the results are complete."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_dir / "requests.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out_dir / "steps.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STEP_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for step in summary["steps"]:
            writer.writerow(step)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
