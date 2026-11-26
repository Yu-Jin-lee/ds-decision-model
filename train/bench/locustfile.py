"""Locust file for trying an endpoint by hand (web UI or headless). The comparable measurement is
bench/job/run.py or bench/endpoint/run.py, which use the same client (decision_model.load_test). This file
talks plain HTTP, so point it at bench/serve_endpoint.py (not at a SageMaker endpoint, which needs SigV4).

    locust -f bench/locustfile.py --host http://127.0.0.1:8080 --records runs/kev-4b/B0-s5-choice/s0/model_inputs_development.jsonl.gz
    locust -f bench/locustfile.py --host http://127.0.0.1:8080 --records ... --headless -u 16 -r 16 -t 2m

--records takes any JSONL(.gz) of records (a run's model_inputs_*.jsonl.gz works; labels are not sent).
The records are sent in a loop, so a long run repeats them (vLLM may answer a repeated prompt from its cache).
--raw-out writes every request (one JSON line each) when the test stops.
"""

from __future__ import annotations

import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from locust import events  # noqa: E402

from decision_model.bench import record_pool  # noqa: E402
from decision_model.load_test import DecisionModelUser, RecordFeed  # noqa: E402,F401  (Locust picks up the user class)


def read_records(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@events.init_command_line_parser.add_listener
def _arguments(parser) -> None:  # noqa: ANN001
    parser.add_argument("--records", required=True, help="JSONL(.gz) of records to send")
    parser.add_argument("--raw-out", default="", help="write every request here (JSONL) when the test stops")


@events.init.add_listener
def _setup(environment, **_kwargs) -> None:  # noqa: ANN001
    environment.record_feed = RecordFeed(record_pool(read_records(environment.parsed_options.records)), cycle=True)
    environment.record_sink = []
    environment.step_start = time.perf_counter()
    environment.step_users = None


@events.quitting.add_listener
def _save(environment, **_kwargs) -> None:  # noqa: ANN001
    path = environment.parsed_options.raw_out
    if path:
        with open(path, "w", encoding="utf-8") as handle:
            for row in environment.record_sink:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
