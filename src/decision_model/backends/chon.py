"""Chon backend: the released Chon typed-decision models (ninhnv/chon-1.0-*), evaluated as is.

Chon is a LoRA adapter on a Gemma 4 instruct model, served by the ``chon`` package
(github.com/xoai/chon) over the same TypeSafe ``/v1/systemone`` wire format as Kev, so
prediction reuses Kev's answer parsing. Only B0 is supported: the release ships no
training recipe, so ``train`` refuses.

Serving defaults keep the comparison with Kev fair: one forward pass per question
(``--no-think``, ``--reread off``), no request deadline, and no yes/no commit band (it
would snap probabilities to 0.2/0.8). The model config's ``serve_args`` replaces them.
Chon still applies its own per-type temperatures from ``chon_config.json``; our
calibration fits one more temperature on top, as for every backend.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .base import Backend, Prediction, strip_labels
from .kev import _free_port, answers_to_log_probs

DEFAULT_SERVE_ARGS = ["--no-think", "--reread", "off", "--deadline", "0", "--noul-commit", "off"]


class ChonBackend(Backend):
    def train(self, records, out_dir: Path, seed: int, replay: int = 0, replay_records=None) -> Path:
        raise NotImplementedError(
            f"{self.config.get('name', 'chon')} is evaluated as released (B0 groups only); Chon ships no training recipe"
        )

    def predict(self, checkpoint: Path | None, records: list[dict[str, Any]]) -> list[Prediction]:
        with self.predictor(checkpoint) as predict_one:
            return [predict_one(record) for record in records]

    def inference_precision(self) -> str:
        return str(self.config.get("dtype", "bfloat16"))

    def serve_command(self, port: int) -> list[str]:
        args = [str(a) for a in self.config.get("serve_args", DEFAULT_SERVE_ARGS)]
        return [sys.executable, "-m", "chon.serve", "--model", self.config["checkpoint"],
                "--dtype", str(self.config.get("dtype", "bfloat16")), "--port", str(port), *args]

    @contextmanager
    def predictor(self, checkpoint: Path | None) -> Iterator[Callable[[dict[str, Any]], Prediction]]:
        """Start ``chon.serve`` once; each call sends one record to ``/v1/systemone``."""
        if checkpoint is not None:
            raise ValueError("Chon has no fine-tuned checkpoints here; it is evaluated as released")
        port = _free_port()
        env = dict(os.environ)
        env.pop("CHON_REREAD", None)
        env.pop("CHON_DEADLINE_S", None)
        server = subprocess.Popen(self.serve_command(port), env=env)
        base_url = f"http://127.0.0.1:{port}"
        try:
            self._wait_ready(base_url, server)

            def predict_one(record: dict[str, Any]) -> Prediction:
                body = {"model": self.config["checkpoint"], "state": record["state"],
                        "questions": strip_labels(record["questions"])}
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
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:   # 400/413 carry the reason in the body
            raise RuntimeError(f"chon.serve answered {error.code}: {error.read()[:500]!r}") from error

    @staticmethod
    def _wait_ready(base_url: str, server: subprocess.Popen, timeout: float = 3600) -> None:
        """The first start downloads the base model and adapter, so wait long."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"chon.serve exited with code {server.returncode}")
            try:
                with urllib.request.urlopen(f"{base_url}/health", timeout=5):
                    return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(5)
        raise TimeoutError("chon.serve did not become ready")
