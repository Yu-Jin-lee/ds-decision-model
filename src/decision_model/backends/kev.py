"""Kev backend: wraps the Kev repository's own trainer and server.

Training runs ``python -m kev.train`` with the recipe in the model config.
Prediction starts ``python -m kev.serve`` on the checkpoint with
``KEV_TEMPERATURE=1.0`` (raw probabilities) and ``KEV_DTYPE=fp32`` (the
evaluation path), then asks every record through ``/v1/systemone``.
Calibration happens afterwards in shared code, as for every backend.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .base import Backend, Prediction, safe_log, strip_labels

DEFAULT_KEV_DIR = Path(os.environ.get("KEV_DIR", "/opt/kev"))

# Set by the training image for its own Python 3.12. Kev's uv project runs its own
# interpreter (3.13 from .python-version); inheriting these makes it load the 3.12
# standard library and fail ("SRE module mismatch").
_HOST_PYTHON_VARS = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")


def uv_environment(**extra: str) -> dict[str, str]:
    """Environment for ``uv`` commands in the Kev checkout, without the host Python's paths."""
    env = {k: v for k, v in os.environ.items() if k not in _HOST_PYTHON_VARS}
    env.update(extra)
    return env


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def answers_to_log_probs(answers: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Convert a /v1/systemone ``answers`` body to per-option log-probabilities."""
    result: dict[str, dict[str, float]] = {}
    for question_id, answer in answers.items():
        if answer["type"] == "noul":
            p = float(answer["noul"])
            result[question_id] = {"true": safe_log(p), "false": safe_log(1.0 - p)}
        else:
            result[question_id] = {str(k): safe_log(float(v)) for k, v in answer["probabilities"].items()}
    return result


class KevBackend(Backend):
    def __init__(self, model_config: dict[str, Any], kev_dir: Path | None = None):
        super().__init__(model_config)
        self.kev_dir = Path(kev_dir or DEFAULT_KEV_DIR)

    def _uv(self, *args: str, extra: str | None = None) -> list[str]:
        command = ["uv", "run"]
        if extra:
            command += ["--extra", extra]
        return command + ["python", "-m", *args]

    def train_command(self, data_path: Path, out_dir: Path, seed: int, replay: int) -> list[str]:
        cfg = self.train_config
        args = [
            "kev.train",
            "--data", str(data_path),
            "--init_from", self.config["checkpoint"],
            "--base", self.config["base"],
            "--lora", str(cfg.get("lora_rank", 16)),
            "--head_dim", str(cfg.get("head_dim", 256)),
            "--lora_targets", str(cfg.get("lora_targets", "all")),
            "--weights_dtype", str(cfg.get("weights_dtype", "fp32")),
            "--epochs", str(cfg.get("epochs", 1)),
            "--lr", str(cfg.get("lr", 2e-5)),
            "--batch", str(cfg.get("batch", 2)),
            "--accum", str(cfg.get("accum", 4)),
            "--checkpointing", str(cfg.get("checkpointing", 1)),
            "--p_none_pair", str(cfg.get("p_none_pair", 0.25)),
            "--seed", str(seed),
            "--device", "cuda",
            "--dtype", str(cfg.get("dtype", "bf16")),
            "--out", str(out_dir),
        ]
        if self.config.get("base_revision"):
            args += ["--base_revision", str(self.config["base_revision"])]
        if replay:
            args += ["--suite", "evals/v7/decision-v7", "--replay", str(replay)]
        args += [str(a) for a in cfg.get("extra_args", [])]
        return self._uv(*args)

    def train(
        self,
        records: list[dict[str, Any]],
        out_dir: Path,
        seed: int,
        replay: int = 0,
        replay_records: list[dict[str, Any]] | None = None,
    ) -> Path:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        data_path = out_dir / "kev_train.jsonl"
        with open(data_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps({"state": record["state"], "questions": record["questions"]}, ensure_ascii=False) + "\n")
        checkpoint = out_dir / "checkpoint"
        subprocess.run(
            self.train_command(data_path, checkpoint, seed, replay), cwd=self.kev_dir, env=uv_environment(), check=True
        )
        return checkpoint

    def predict(self, checkpoint: Path | None, records: list[dict[str, Any]]) -> list[Prediction]:
        with self.predictor(checkpoint) as predict_one:
            return [predict_one(record) for record in records]

    def inference_precision(self) -> str:
        return "fp32 (KEV_DTYPE=fp32, evaluation path)"

    @contextmanager
    def predictor(self, checkpoint: Path | None) -> Iterator[Callable[[dict[str, Any]], Prediction]]:
        """Start ``kev.serve`` once; each call sends one record to ``/v1/systemone``.

        Timing through the local server includes a localhost HTTP round trip,
        which is how Kev serves in production.
        """
        port = _free_port()
        run = str(checkpoint) if checkpoint is not None else self.config["checkpoint"]
        env = uv_environment(KEV_TEMPERATURE="1.0", KEV_DTYPE="fp32")
        server = subprocess.Popen(
            self._uv("kev.serve", "--run", run, "--port", str(port), extra="serve"),
            cwd=self.kev_dir,
            env=env,
        )
        base_url = f"http://127.0.0.1:{port}"
        try:
            self._wait_ready(base_url, server)

            def predict_one(record: dict[str, Any]) -> Prediction:
                body = {"model": "kev-latest", "state": record["state"], "questions": strip_labels(record["questions"])}
                response = self._post(f"{base_url}/v1/systemone", body)
                return {"id": record["id"], "answers": answers_to_log_probs(response["answers"])}

            yield predict_one
        finally:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()

    @staticmethod
    def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())

    @staticmethod
    def _wait_ready(base_url: str, server: subprocess.Popen, timeout: float = 1800) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"kev.serve exited with code {server.returncode}")
            try:
                with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5):
                    return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(5)
        raise TimeoutError("kev.serve did not become ready")
