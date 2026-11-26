#!/usr/bin/env python3
"""Serve one model as an HTTP endpoint with its serving profile (configs/serving.yaml), for the load test.

    python bench/serve_endpoint.py --model kev-4b --port 8080
    python bench/serve_endpoint.py --model laya-multilingual --checkpoint runs/.../checkpoint --port 8080

POST /predict (= /invocations) takes one record and answers its prediction; GET /health (= /ping), GET /info
(decision_model.endpoint). Both methods use it: bench/job/run.py starts it on localhost, and the image of
bench/endpoint/ starts it on port 8080 inside the SageMaker endpoint.
"""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_model.backends import get_backend  # noqa: E402
from decision_model.config import load_model_config  # noqa: E402
from decision_model.endpoint import make_server, open_endpoint  # noqa: E402
from decision_model.latency import environment  # noqa: E402


def _stop(signum, frame):  # noqa: ANN001
    # SIGTERM -> leave serve_forever through the context managers, so the backend stops its own servers
    # (kev.serve, chon.serve, vLLM) instead of leaving them running.
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="config name under configs/models")
    parser.add_argument("--checkpoint", default=None, help="a trained run's checkpoint (default: the released model)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-batch", type=int, default=None,
                        help="records per forward pass for a model run in this process (default: the model "
                             "config's serving.endpoint; 1 = no batching)")
    parser.add_argument("--max-wait-ms", type=float, default=None, help="wait this long for a batch to fill")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    backend = get_backend(load_model_config(args.model), serving=True)
    checkpoint = Path(args.checkpoint) if args.checkpoint else None
    with open_endpoint(backend, checkpoint, args.max_batch, args.max_wait_ms) as endpoint:
        endpoint.info["environment"] = environment()
        server = make_server(endpoint, args.host, args.port)
        print(f"endpoint ready on http://{args.host}:{args.port} ({endpoint.info['mode']})", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()


if __name__ == "__main__":
    main()
