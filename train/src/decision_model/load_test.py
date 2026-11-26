"""Locust side of the endpoint load test: the simulated clients and the closed-loop steps.

Two clients send the same records and record the same row per request:

* DecisionModelUser: plain HTTP to ``POST /predict`` (method ``job``: the endpoint on the same machine).
* SageMakerEndpointUser: ``InvokeEndpoint`` on a SageMaker real-time endpoint, signed with SigV4 (method
  ``endpoint``). One boto3 client per user, no retries, so throttling and errors show up as failures.

Importing this module imports Locust, which patches the standard library for gevent; import it only in the
load-test process (bench/job/run.py, bench/endpoint/run.py, bench/locustfile.py), never in the endpoint.
"""

from __future__ import annotations

import itertools
import json
import logging
import resource
import time
from typing import Any, Callable, Iterable

import gevent
from locust import FastHttpUser, User, constant, task
from locust.env import Environment
from locust.exception import StopUser

from .bench import summarize_step

# Every user of a step starts at once on purpose (the ramp is left out of the numbers, decision_model.bench);
# Locust warns about that spawn rate on every step.
logging.getLogger("locust.runners").addFilter(lambda record: "spawn rate is very high" not in record.getMessage())


class RecordFeed:
    """Hands each request its record. gevent runs one greenlet at a time, so no lock is needed."""

    def __init__(self, records: Iterable[dict[str, Any]], cycle: bool = False):
        items = [(r, json.dumps(r, ensure_ascii=False).encode("utf-8")) for r in records]   # encode once
        if not items:
            raise ValueError("no records to send")
        self.total = len(items)
        self._items = itertools.cycle(items) if cycle else iter(items)

    def next(self) -> tuple[dict[str, Any], bytes] | None:
        return next(self._items, None)


def _new_row(environment: Environment, record: dict[str, Any]) -> dict[str, Any]:
    return {"id": record["id"], "task": record["task"], "lang": record["lang"],
            "questions": len(record["questions"]), "users": getattr(environment, "step_users", None)}


def _check_answer(row: dict[str, Any], record: dict[str, Any], status: int, content: bytes | None) -> bool:
    """Fill the row from the endpoint's answer; True when it is this record's prediction."""
    row["status"] = status
    if status != 200 or not content:
        return False
    try:
        answer = json.loads(content)
        row.update(server_ms=answer.get("server_ms"), batch=answer.get("batch"))
        prediction = answer.get("prediction")
        if not (isinstance(prediction, dict) and prediction.get("id") == record["id"]):
            return False
        # Each question's most likely option: lets a run be checked against one-request-at-a-time answers
        # (a server that batches must answer the same).
        row["top"] = {qid: max(options, key=options.get) for qid, options in (prediction.get("answers") or {}).items()
                      if isinstance(options, dict) and options}
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _timing(row: dict[str, Any], environment: Environment, start: float, end: float) -> None:
    row.update(start_s=round(start - environment.step_start, 6), end_s=round(end - environment.step_start, 6),
               ms=round((end - start) * 1000.0, 3))


class DecisionModelUser(FastHttpUser):
    """Sends one record, waits for the answer, sends the next (closed loop, no think time).

    Expects ``environment.record_feed`` (a RecordFeed), ``environment.record_sink`` (a list that gets one row
    per request) and ``environment.step_start`` (perf_counter at the start of the step)."""

    wait_time = constant(0)
    network_timeout = 600.0
    connection_timeout = 60.0

    @task
    def predict(self) -> None:
        environment = self.environment
        item = environment.record_feed.next()
        if item is None:
            raise StopUser()
        record, body = item
        row = _new_row(environment, record)
        start = time.perf_counter()
        with self.client.post("/predict", data=body, headers={"content-type": "application/json"},
                              name=f"/predict {record['task']}", catch_response=True) as response:
            end = time.perf_counter()
            _timing(row, environment, start, end)
            ok = _check_answer(row, record, response.status_code, response.content)
            if ok:
                response.success()
            else:
                text = (str(response.error) if response.error else (response.text or ""))[:300]
                row["error"] = text
                response.failure(f"{response.status_code}: {text}")
            row["ok"] = ok
        environment.record_sink.append(row)


class SageMakerEndpointUser(User):
    """Like DecisionModelUser, through ``sagemaker-runtime:InvokeEndpoint``.

    Expects ``environment.endpoint_name`` and ``environment.region`` as well."""

    wait_time = constant(0)

    def __init__(self, environment: Environment):
        super().__init__(environment)
        import boto3
        from botocore.config import Config

        config = Config(retries={"max_attempts": 1, "mode": "standard"}, read_timeout=600, connect_timeout=60,
                        max_pool_connections=2)
        self.runtime = boto3.session.Session().client("sagemaker-runtime", region_name=environment.region,
                                                      config=config)

    @task
    def predict(self) -> None:
        environment = self.environment
        item = environment.record_feed.next()
        if item is None:
            raise StopUser()
        record, body = item
        row = _new_row(environment, record)
        start = time.perf_counter()
        error: Exception | None = None
        status, content = 0, None
        try:
            response = self.runtime.invoke_endpoint(EndpointName=environment.endpoint_name, Body=body,
                                                    ContentType="application/json", Accept="application/json")
            status = response["ResponseMetadata"]["HTTPStatusCode"]
            content = response["Body"].read()
        except Exception as caught:  # noqa: BLE001 - throttling, model errors, timeouts: a failed request
            error = caught
            status = getattr(caught, "response", {}).get("ResponseMetadata", {}).get("HTTPStatusCode", 0)
        end = time.perf_counter()
        _timing(row, environment, start, end)
        ok = error is None and _check_answer(row, record, status, content)
        if not ok:
            row["error"] = (f"{type(error).__name__}: {error}" if error else (content or b"")[:300].decode("utf-8", "replace"))[:300]
        row["ok"] = ok
        environment.events.request.fire(
            request_type="InvokeEndpoint", name=f"/invocations {record['task']}", response_time=row["ms"],
            response_length=len(content or b""), exception=None if ok else RuntimeError(row.get("error")),
            context={}, start_time=time.time() - row["ms"] / 1000.0,
        )
        environment.record_sink.append(row)


def run_step(records: list[dict[str, Any]], users: int, max_seconds: float, user_class: type[User] = DecisionModelUser,
             host: str | None = None, **environment_attributes: Any) -> dict[str, Any]:
    """One closed-loop step: ``users`` clients until every record has an answer (or ``max_seconds``)."""
    environment = Environment(user_classes=[user_class], host=host)
    for key, value in environment_attributes.items():
        setattr(environment, key, value)
    environment.record_feed = RecordFeed(records)
    environment.record_sink = []
    environment.step_users = users
    runner = environment.create_local_runner()
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
    environment.step_start = time.perf_counter()
    runner.start(min(users, len(records)), spawn_rate=10_000)
    timed_out = False
    while len(environment.record_sink) < len(records):
        if time.perf_counter() - environment.step_start > max_seconds:
            timed_out = True
            break
        gevent.sleep(0.02)
    seconds = time.perf_counter() - environment.step_start
    cpu_after = resource.getrusage(resource.RUSAGE_SELF)
    runner.quit()
    cpu = (cpu_after.ru_utime - cpu_before.ru_utime) + (cpu_after.ru_stime - cpu_before.ru_stime)
    return {"rows": list(environment.record_sink), "seconds": seconds, "timed_out": timed_out,
            "client_cpu": cpu / seconds if seconds > 0 else None}


def run_sweep(plan: dict[str, Any], max_step_seconds: float, user_class: type[User] = DecisionModelUser,
              host: str | None = None, log: Callable[[str], None] = print,
              **environment_attributes: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Warm-up (half with one user, half with the most users), then every step of the plan
    (decision_model.bench.plan_steps). Returns (step summaries, every request)."""
    def step(records: list[dict[str, Any]], users: int) -> dict[str, Any]:
        return run_step(records, users, max_step_seconds, user_class, host, **environment_attributes)

    warm = plan["warmup"]
    most_users = max(s["users"] for s in plan["steps"])
    if warm:
        half = len(warm) // 2
        step(warm[:half] or warm, 1)
        if warm[half:]:
            step(warm[half:], most_users)

    steps: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for planned in plan["steps"]:
        result = step(planned["records"], planned["users"])
        stats = summarize_step(result["rows"], planned["users"], result["seconds"], result["client_cpu"],
                               result["timed_out"])
        steps.append(stats)
        rows += result["rows"]
        log(f"users {stats['users']:>3}: {stats['records_per_s']} records/s, p50 {stats['p50_ms']} ms, "
            f"p95 {stats['p95_ms']} ms, failed {stats['failed']}, batch {stats['mean_batch']}, "
            f"client cpu {stats['client_cpu']}")
        if result["timed_out"]:
            log(f"step with {planned['users']} users hit max_step_seconds; stopping here")
            break
    return steps, rows
