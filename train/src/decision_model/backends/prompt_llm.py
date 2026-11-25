"""Generative LLM answered through its production prompt (e.g. Gemma 4), as a baseline.

The model is not fine-tuned. Each task question is sent as the team's production prompt
(one file per task, variant and language) to a local vLLM server, the model generates
text as in production (greedy), and :func:`parse_answer` reads the leading number
(``2``, ``1|samsung``, ``7|history``) and maps it to this repo's label. An output that
does not parse counts as wrong. There are no probabilities, so only accuracy, the
invalid-output rate and latency are reported (no temperature, Brier, ECE or auto rate).
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from ..formats import INVALID
from .base import Backend, Prediction

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VLLM_VENV = Path(os.environ.get("VLLM_VENV", "/opt/vllm-venv"))
FLOOR_LOG_PROB = math.log(1e-12)


def render_serp(serp: Any) -> str:
    """The assembled SERP state (AI overview, or top results) as plain text."""
    if serp is None:
        return ""
    if isinstance(serp, str):
        return serp
    if isinstance(serp, dict):
        if serp.get("ai_overview"):
            return str(serp["ai_overview"])
        lines = []
        for result in serp.get("results") or []:
            title, snippet = result.get("title", ""), result.get("snippet", "")
            lines.append(f"- {title}: {snippet}" if title and snippet else f"- {title or snippet}")
        return "\n".join(lines)
    return json.dumps(serp, ensure_ascii=False)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def prompt_fields(task: str, state: dict[str, Any]) -> dict[str, str]:
    """Values for the production prompt's placeholders, from the same state the other models see."""
    serp = render_serp(state.get("serp"))
    if task == "s2":
        market = " > ".join(p for p in (_text(state.get("industry")), _text(state.get("category"))) if p)
        return {"market": market, "related_entities": _text(state.get("seeds")),
                "keyword": _text(state.get("keyword")), "serp_snippet": serp}
    if task == "s4":
        return {"keyword": _text(state.get("keyword")), "target": _text(state.get("target")), "serp": serp}
    if task == "s5":
        return {name: _text(state.get(name)) for name in ("keyword", "target", "intent", "site")} | {"serp": serp}
    raise ValueError(f"no production prompt for task {task!r}")


def fill(template: str, fields: dict[str, str]) -> str:
    """Replace only the known ``{name}`` placeholders (prompts contain other braces as text)."""
    for name, value in fields.items():
        template = template.replace("{" + name + "}", value)
    return template


_LEADING_NUMBER = re.compile(r"^[\s*`'\"\[(]*(\d+)")


def parse_answer(text: str, digits: dict[str, str]) -> str:
    """Label from a generated answer: the leading number before any ``|`` text, as the
    production prompts specify (``2``, ``1|samsung``, ``7|history``). INVALID if there is
    no leading number or it is not one of the prompt's answers."""
    match = _LEADING_NUMBER.match(text or "")
    if not match:
        return INVALID
    return digits.get(str(int(match.group(1))), INVALID)


def one_hot_log_probs(label: str, labels: list[str]) -> dict[str, float]:
    """The parsed answer as a log-probability map over the labels (INVALID gets its own key)."""
    out = {name: FLOOR_LOG_PROB for name in labels}
    out[label] = 0.0
    return out


class PromptLLMBackend(Backend):
    generative = True   # parsed text answers: no calibration, no probability metrics

    def __init__(self, model_config: dict[str, Any], venv: Path | None = None, repo_root: Path = REPO_ROOT):
        super().__init__(model_config)
        self.venv = Path(venv or DEFAULT_VLLM_VENV)
        self.vllm = dict(model_config.get("vllm") or {})
        self.prompt_dir = repo_root / model_config["prompt_dir"]
        self._templates: dict[str, str] = {}

    # -- prompts -------------------------------------------------------------------------
    def prompt_spec(self, task: str, variant: str | None) -> dict[str, Any]:
        spec = self.config["prompts"][task]
        if "file" not in spec:
            if variant not in spec:
                raise ValueError(f"{task}: no production prompt for variant {variant!r}")
            spec = spec[variant]
        return spec

    def template(self, file: str, lang: str) -> str:
        name = f"{file}_{self.config['lang_suffix'][lang]}.txt"
        if name not in self._templates:
            self._templates[name] = (self.prompt_dir / name).read_text(encoding="utf-8")
        return self._templates[name]

    def build_prompt(self, record: dict[str, Any]) -> tuple[str, dict[str, str]]:
        spec = self.prompt_spec(record["task"], record.get("variant"))
        prompt = fill(self.template(spec["file"], record["lang"]), prompt_fields(record["task"], record["state"]))
        return prompt, spec["digits"]

    @staticmethod
    def generated_question(record: dict[str, Any]) -> str:
        questions = record["questions"]
        ids = [qid for qid, q in questions.items() if q.get("type") == "generated"]
        if len(questions) != 1 or len(ids) != 1:
            raise ValueError(f"{record['id']}: a prompt baseline answers its production prompt only; run a P0-* group")
        return ids[0]

    # -- Backend -------------------------------------------------------------------------
    def train(self, records, out_dir, seed, replay=0, replay_records=None):  # noqa: ANN001
        raise NotImplementedError(f"{self.config['name']} is a reference baseline and is not fine-tuned; use a P0-* group")

    def vllm_args(self) -> list[str]:
        """vLLM server flags: the serving profile's ``vllm_args`` when it gives them, else the evaluation ones."""
        return [str(a) for a in (self.serving_config.get("vllm_args") or self.vllm.get("args", []))]

    def inference_precision(self) -> str:
        args = self.vllm_args()
        quant = args[args.index("--quantization") + 1] if "--quantization" in args else "none"
        dtype = args[args.index("--dtype") + 1] if "--dtype" in args else "auto"
        return f"{dtype} compute, {quant} weights (vLLM {self.vllm_version()})"

    def vllm_version(self) -> str:
        try:
            return subprocess.check_output(
                [str(self.venv / "bin" / "python"), "-c", "import vllm; print(vllm.__version__)"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unknown"

    def predict(self, checkpoint: Path | None, records: list[dict[str, Any]]) -> list[Prediction]:
        with self.predictor(checkpoint) as predict_one:
            workers = int(self.vllm.get("eval_concurrency", 16))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(predict_one, records))

    @contextmanager
    def predictor(self, checkpoint: Path | None) -> Iterator[Callable[[dict[str, Any]], Prediction]]:
        """Start one vLLM server; each call sends one record's prompt to /v1/chat/completions."""
        if checkpoint is not None:
            raise ValueError(f"{self.config['name']} has no fine-tuned checkpoints")
        port = _free_port()
        # A local copy (resharded by sagemaker/entry.py) if there is one, served under the checkpoint's name.
        model_path = os.environ.get("VLLM_MODEL_PATH") or self.config["checkpoint"]
        command = [str(self.venv / "bin" / "vllm"), "serve", model_path,
                   "--served-model-name", self.config["checkpoint"],
                   "--port", str(port), "--host", "127.0.0.1", *self.vllm_args()]
        print("+", " ".join(command), flush=True)
        server = subprocess.Popen(command, env=_clean_env())
        base_url = f"http://127.0.0.1:{port}"
        max_tokens = int(self.vllm.get("max_tokens", 64))
        try:
            _wait_ready(base_url, server)

            def predict_one(record: dict[str, Any]) -> Prediction:
                qid = self.generated_question(record)
                prompt, digits = self.build_prompt(record)
                body = {
                    "model": self.config["checkpoint"],
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                }
                response = _post(f"{base_url}/v1/chat/completions", body)
                text = response["choices"][0]["message"].get("content") or ""
                label = parse_answer(text, digits)
                return {"id": record["id"], "answers": {qid: one_hot_log_probs(label, sorted(set(digits.values())))},
                        "output": text, "parsed": label}

            yield predict_one
        finally:
            server.terminate()
            try:
                server.wait(timeout=120)
            except subprocess.TimeoutExpired:
                server.kill()


def _clean_env() -> dict[str, str]:
    """The vLLM venv brings its own torch; keep the training image's Python paths out."""
    drop = ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_PREFIX")
    env = {k: v for k, v in os.environ.items() if k not in drop}
    # ~52 GB of weights: keep the Hugging Face cache on the job's data volume, not the container disk.
    env.setdefault("HF_HOME", os.environ.get("VLLM_HF_HOME", "/opt/ml/hf_cache"))
    # vLLM's FlashInfer sampler JIT-compiles its kernels at engine start and needs the CUDA toolkit headers
    # (curand.h), which the vLLM venv does not have: the 2026-11-23 serving job's engine stopped there. The
    # answers are greedy one-line numbers, so the PyTorch sampler gives the same output.
    env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return env


def _free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read())


def _wait_ready(base_url: str, server: subprocess.Popen, timeout: float = 3600) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vllm serve exited with code {server.returncode}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5):
                return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(5)
    raise TimeoutError("vllm serve did not become ready")
