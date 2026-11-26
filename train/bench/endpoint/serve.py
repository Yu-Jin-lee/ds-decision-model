#!/usr/bin/env python3
"""Entrypoint of the endpoint image (bench/endpoint/Dockerfile): get the weights, then serve on port 8080.

Environment (set by bench/endpoint/deploy.py on the SageMaker model):

    DECISION_MODEL_NAME   model config name (baked into the image)
    BENCH_GROUP, BENCH_SEED  the run whose checkpoint to serve; a trained run downloads its checkpoint from
                          SM_OUTPUT_PREFIX (the runs prefix), B0 and P0 serve the released model
    SM_OUTPUT_PREFIX      S3 runs prefix (only needed for trained runs)

Gemma (prompt_llm) is downloaded and resharded first, as in the serving job. Until the server listens,
/ping fails and SageMaker keeps waiting (configs/bench.yaml endpoint.startup_timeout_s).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from decision_model.config import load_model_config  # noqa: E402


def _entry():
    spec = importlib.util.spec_from_file_location("entry", REPO / "sagemaker" / "entry.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def server_command(model: str, checkpoint_args: list[str]) -> list[str]:
    return [sys.executable, str(REPO / "bench" / "serve_endpoint.py"), "--model", model,
            "--host", "0.0.0.0", "--port", "8080", *checkpoint_args]


def main() -> None:
    model = os.environ["DECISION_MODEL_NAME"]
    group, seed = os.environ.get("BENCH_GROUP"), int(os.environ.get("BENCH_SEED", "0"))
    entry = _entry()
    config = load_model_config(model)
    if config["backend"] == "prompt_llm":
        entry.prepare_weights(config)   # sets VLLM_MODEL_PATH, inherited by the server
    checkpoint_args = entry.serving_checkpoint_args(model, group, seed) if group else []
    command = server_command(model, checkpoint_args)
    print("+", " ".join(command), flush=True)
    os.execv(command[0], command)   # the server gets SageMaker's signals directly


if __name__ == "__main__":
    main()
