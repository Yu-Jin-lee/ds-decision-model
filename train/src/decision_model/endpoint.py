"""One HTTP endpoint for every model, for the load test (configs/bench.yaml, train/bench/).

    POST /predict   body: one record (id, task, lang, state, questions; labels are ignored)
                    answer: {"prediction": {...}, "server_ms": ..., "batch": n}
    GET  /health    200 once the model is loaded
    GET  /info      model, precision, how requests are run (concurrent / batch / serial)

SageMaker real-time endpoints call the same handlers as ``GET /ping`` and ``POST /invocations`` on port 8080
(method ``endpoint``, bench/endpoint/). SageMaker forwards only those two, so a body of ``{"__info__": true}``
answers what ``GET /info`` does.

The prediction comes from the backend's own ``predictor`` (or ``batch_predictor``), the same function the
evaluation and latency runs use, so the model gets exactly the input production gives it. How concurrent
requests reach the model:

* ``concurrent``: the backend only forwards to a server it started (kev.serve, chon.serve, vLLM), so every
  request goes straight through and that server batches them as it does in production.
* ``batch``: a model in this process with a batch predictor (Laya). One worker takes the requests waiting in
  the queue, up to ``max_batch`` records, and runs them in one call (``max_wait_ms`` > 0 also waits that
  long for more to arrive).
* ``serial``: a model in this process without one: one request at a time.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Iterator

from .backends.base import Backend, Prediction


class MicroBatcher:
    """Runs ``predict_many`` on the requests that are waiting, one batch at a time, in one worker thread."""

    def __init__(self, predict_many: Callable[[list[dict[str, Any]]], list[Prediction]], max_batch: int,
                 max_wait_ms: float = 0.0):
        if max_batch < 1:
            raise ValueError("max_batch must be at least 1")
        self.predict_many = predict_many
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000.0
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._worker = threading.Thread(target=self._loop, name="micro-batcher", daemon=True)
        self._worker.start()

    def predict(self, record: dict[str, Any]) -> tuple[Prediction, int]:
        """(prediction, size of the batch it ran in)."""
        if self._closed:
            raise RuntimeError("batcher is closed")
        future: Future = Future()
        self._queue.put((record, future))
        return future.result()

    def close(self) -> None:
        self._closed = True
        self._queue.put(None)
        self._worker.join(timeout=10)

    def _take(self) -> list[tuple[dict[str, Any], Future]] | None:
        first = self._queue.get()
        if first is None:
            return None
        batch = [first]
        deadline = time.monotonic() + self.max_wait
        while len(batch) < self.max_batch:
            try:
                remaining = deadline - time.monotonic()
                item = self._queue.get(timeout=remaining) if remaining > 0 else self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._queue.put(None)   # finish this batch, then stop
                break
            batch.append(item)
        return batch

    def _loop(self) -> None:
        while True:
            batch = self._take()
            if batch is None:
                return
            try:
                predictions = self.predict_many([record for record, _ in batch])
                if len(predictions) != len(batch):
                    raise RuntimeError(f"batch predictor returned {len(predictions)} predictions for {len(batch)} records")
                for (_, future), prediction in zip(batch, predictions):
                    future.set_result((prediction, len(batch)))
            except BaseException as error:  # noqa: BLE001 - every waiting request gets the error
                for _, future in batch:
                    if not future.done():
                        future.set_exception(error)


class Endpoint:
    """Turns one record into one prediction, whatever the backend's concurrency (see the module docstring)."""

    def __init__(self, mode: str, predict_one: Callable[[dict[str, Any]], Prediction] | None = None,
                 batcher: MicroBatcher | None = None, info: dict[str, Any] | None = None):
        if mode not in ("concurrent", "batch", "serial"):
            raise ValueError(f"unknown mode {mode!r}")
        if (mode == "batch") != (batcher is not None) or (mode != "batch" and predict_one is None):
            raise ValueError(f"mode {mode!r} needs {'a batcher' if mode == 'batch' else 'predict_one'}")
        self.mode = mode
        self.predict_one = predict_one
        self.batcher = batcher
        self.info = dict(info or {})
        self._lock = threading.Lock()

    def predict(self, record: dict[str, Any]) -> tuple[Prediction, int]:
        if self.mode == "batch":
            return self.batcher.predict(record)
        if self.mode == "serial":
            with self._lock:
                return self.predict_one(record), 1
        return self.predict_one(record), 1


def endpoint_settings(model_config: dict[str, Any]) -> dict[str, Any]:
    """The serving profile's ``endpoint`` section (max_batch, max_wait_ms), empty when it has none."""
    return dict((model_config.get("serving") or {}).get("endpoint") or {})


@contextmanager
def open_endpoint(backend: Backend, checkpoint: Path | None, max_batch: int | None = None,
                  max_wait_ms: float | None = None) -> Iterator[Endpoint]:
    """Load the model once and yield an Endpoint. ``max_batch``/``max_wait_ms`` default to the model config's
    ``serving.endpoint``; ``max_batch`` 1 turns batching off."""
    settings = endpoint_settings(backend.config)
    max_batch = int(max_batch if max_batch is not None else settings.get("max_batch", 1))
    max_wait_ms = float(max_wait_ms if max_wait_ms is not None else settings.get("max_wait_ms", 0))
    info = {
        "model": backend.config.get("name"),
        "backend": backend.config.get("backend"),
        "checkpoint": str(checkpoint) if checkpoint is not None else backend.config.get("checkpoint"),
        "profile": "serving" if backend.serving else "evaluation",
        "precision": backend.inference_precision(),
    }
    with ExitStack() as stack:
        if backend.concurrent_predictor:
            endpoint = Endpoint("concurrent", predict_one=stack.enter_context(backend.predictor(checkpoint)), info=info)
        elif max_batch > 1 and backend.has_batch_predictor():
            batcher = MicroBatcher(stack.enter_context(backend.batch_predictor(checkpoint)), max_batch, max_wait_ms)
            stack.callback(batcher.close)
            endpoint = Endpoint("batch", batcher=batcher, info=info)
        else:
            endpoint = Endpoint("serial", predict_one=stack.enter_context(backend.predictor(checkpoint)), info=info)
        endpoint.info.update({"mode": endpoint.mode,
                              "max_batch": max_batch if endpoint.mode == "batch" else 1,
                              "max_wait_ms": max_wait_ms if endpoint.mode == "batch" else 0.0})
        yield endpoint


def make_server(endpoint: Endpoint, host: str = "127.0.0.1", port: int = 8080) -> ThreadingHTTPServer:
    """A threaded HTTP/1.1 server (keep-alive, so the load test does not open a connection per request)."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - quiet under load
            pass

        def _send(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path in ("/health", "/ping"):
                self._send(200, {"status": "ok"})
            elif self.path == "/info":
                self._send(200, endpoint.info)
            else:
                self._send(404, {"error": f"no route {self.path}"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in ("/predict", "/invocations"):
                self._send(404, {"error": f"no route {self.path}"})
                return
            length = int(self.headers.get("content-length") or 0)
            try:
                record = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError) as error:
                self._send(400, {"error": f"body is not JSON: {error}"})
                return
            if isinstance(record, dict) and record.get("__info__"):
                self._send(200, endpoint.info)
                return
            start = time.perf_counter()
            try:
                prediction, batch = endpoint.predict(record)
            except Exception as error:  # noqa: BLE001 - the client counts it as a failed request
                self._send(500, {"error": f"{type(error).__name__}: {error}"[:2000]})
                return
            server_ms = (time.perf_counter() - start) * 1000.0
            self._send(200, {"prediction": prediction, "server_ms": round(server_ms, 3), "batch": batch})

    return _Server((host, port), Handler)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512   # listen backlog: the load test opens up to max(users) connections at once
