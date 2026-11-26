#!/usr/bin/env python3
"""Method ``job``: serve one model on this machine and load-test it with Locust over localhost.

Runs inside the SageMaker job that bench/job/launch.py submits, or on any GPU machine. Starts the endpoint
itself (bench/serve_endpoint.py, serving profile), or measures one already running (--url). Settings come
from configs/bench.yaml; flags override them.

    python bench/job/run.py --model kev-4b --group B0-s5-choice --seed 0
    python bench/job/run.py --model laya-multilingual --group S-s5-choice --seed 0 --checkpoint <checkpoint>
    python bench/job/run.py --model kev-4b --group B0-s5-choice --url http://127.0.0.1:8080

Results: {out-root}/bench/job/{model}/{group}/s{seed}/ (with --label, one folder deeper):
summary.json, steps.csv, requests.jsonl.gz. Compare with bench/compare.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# Locust patches the standard library for gevent; import it before anything opens sockets.
from decision_model.load_test import DecisionModelUser, run_sweep  # noqa: E402,I001

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from typing import Any  # noqa: E402

from decision_model.bench import (  # noqa: E402
    build_plan, describe_plan, load_bench_config, make_summary, result_dir, write_results,
)
from decision_model.pipeline import find_run  # noqa: E402


def _get(url: str, timeout: float = 10) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def wait_ready(url: str, server: subprocess.Popen | None, timeout: float) -> None:
    """The first start downloads the model, so wait long."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server is not None and server.poll() is not None:
            raise RuntimeError(f"endpoint exited with code {server.returncode}")
        try:
            _get(f"{url}/health", timeout=5)
            return
        except (urllib.error.URLError, ConnectionError, OSError, ValueError):
            time.sleep(5)
    raise TimeoutError(f"endpoint at {url} did not become ready")


def start_endpoint(args: argparse.Namespace) -> tuple[subprocess.Popen, str]:
    command = [sys.executable, str(REPO / "bench" / "serve_endpoint.py"), "--model", args.model,
               "--host", "127.0.0.1", "--port", str(args.port)]
    if args.checkpoint:
        command += ["--checkpoint", args.checkpoint]
    if args.max_batch is not None:
        command += ["--max-batch", str(args.max_batch)]
    if args.max_wait_ms is not None:
        command += ["--max-wait-ms", str(args.max_wait_ms)]
    print("+", " ".join(command), flush=True)
    return subprocess.Popen(command), f"http://127.0.0.1:{args.port}"


def stop_endpoint(server: subprocess.Popen) -> None:
    server.terminate()
    try:
        server.wait(timeout=180)
    except subprocess.TimeoutExpired:
        server.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--group", required=True, help="expanded group name; picks the task(s), level and languages")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None, help="a trained run's checkpoint (default: the released model)")
    parser.add_argument("--url", default=None, help="measure an endpoint that is already running")
    parser.add_argument("--port", type=int, default=8080, help="port for the endpoint this script starts")
    parser.add_argument("--max-batch", type=int, default=None, help="passed to serve_endpoint.py")
    parser.add_argument("--max-wait-ms", type=float, default=None, help="passed to serve_endpoint.py")
    parser.add_argument("--data-root", default=os.environ.get("SM_CHANNEL_DATA", "data"))
    parser.add_argument("--out-root", default=os.environ.get("SM_MODEL_DIR", "runs"))
    parser.add_argument("--label", default=None, help="variant name, e.g. batch1 (results one folder deeper)")
    parser.add_argument("--users", default=None, help="comma-separated, e.g. 1,4,16 (default: configs/bench.yaml)")
    parser.add_argument("--requests-per-step", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--ready-timeout", type=float, default=3600)
    args = parser.parse_args()

    config = load_bench_config()
    run = find_run(args.model, args.group, args.seed)
    plan = build_plan(run, args.data_root, config, [int(u) for u in args.users.split(",")] if args.users else None,
                      args.requests_per_step, args.warmup)
    print(describe_plan(run.run_id, plan), flush=True)

    server = None
    started = time.time()
    try:
        if args.url:
            url = args.url.rstrip("/")
        else:
            server, url = start_endpoint(args)
        wait_ready(url, server, args.ready_timeout)
        info = _get(f"{url}/info")
        print(f"endpoint: {json.dumps({k: v for k, v in info.items() if k != 'environment'}, ensure_ascii=False)}",
              flush=True)
        steps, rows = run_sweep(plan, config["max_step_seconds"], DecisionModelUser, host=url,
                                log=lambda line: print(line, flush=True))
    finally:
        if server is not None:
            stop_endpoint(server)

    summary = make_summary("job", run, plan, steps, info, config, args.label, args.checkpoint,
                           url=url if args.url else None, client_on_endpoint_host=not args.url,
                           seconds=round(time.time() - started, 1))
    out_dir = result_dir(args.out_root, "job", run.run_id, args.label)
    write_results(out_dir, summary, rows)
    print(f"wrote {out_dir}", flush=True)
    print(json.dumps(summary["best"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
