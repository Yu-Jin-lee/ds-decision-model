"""``chon.serve`` that takes concurrent requests, so its vLLM reader can batch them (serving profile).

``chon.serve`` holds one lock around every request ("requests are served one at a time on the GPU ... a lock keeps
measurements clean"), so under load its vLLM reader only ever sees one request and cannot batch. kev.serve batches
concurrent requests itself and vLLM batches Gemma's, so for the load test (configs/bench.yaml) every model must be
able to batch. Two changes, made before chon.serve starts and nothing else:

* the lock in ``create_app`` becomes a no-op, so requests run at the same time (FastAPI threads, at most
  ``THREADS``);
* the four fields ``DecisionModel.predict`` keeps on the model for the request it is answering (``_req_layout``,
  ``_req_asked``, ``_deadline``, ``_shared``) become per thread. ``predict`` sets them at the start of a request and
  clears them at its end, so shared between threads one request's end would clear another's prompt layout.

A single request runs exactly as before, so a run with one request at a time (latency, concurrency 1) is unchanged.
Run as ``python -m decision_model.backends.chon_serve <chon.serve arguments>``.
"""

from __future__ import annotations

import threading
import types

PER_REQUEST_FIELDS = ("_req_layout", "_req_asked", "_deadline", "_shared")
THREADS = 128   # FastAPI's default thread pool is 40; the load test sends up to 64 requests at once


class _NoLock:
    def __enter__(self) -> "_NoLock":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False


def _per_thread(name: str, local: threading.local) -> property:
    # Every field starts as None in DecisionModel.__init__, so a thread that has not set it yet reads None too.
    return property(lambda self: getattr(local, name, None), lambda self, value: setattr(local, name, value))


def patch() -> None:
    from chon import infer, serve

    local = threading.local()
    for name in PER_REQUEST_FIELDS:
        setattr(infer.DecisionModel, name, _per_thread(name, local))
    serve.threading = types.SimpleNamespace(Lock=_NoLock)   # create_app's lock is chon.serve's only use of threading

    create_app = serve.create_app

    def create_app_concurrent(*args, **kwargs):
        app = create_app(*args, **kwargs)

        @app.on_event("startup")
        async def _threads() -> None:
            import anyio.to_thread

            anyio.to_thread.current_default_thread_limiter().total_tokens = THREADS

        return app

    serve.create_app = create_app_concurrent


def main() -> None:
    patch()
    from chon import serve

    serve.main()


if __name__ == "__main__":
    main()
