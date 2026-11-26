"""Chon backend: the released Chon typed-decision models (ninhnv/chon-1.0-*), evaluated as is.

Chon is a LoRA adapter on a Gemma 4 instruct model, served by the ``chon`` package
(github.com/xoai/chon) over the same TypeSafe ``/v1/systemone`` wire format as Kev, so
prediction reuses Kev's answer parsing.

Training (``train``): the release ships no training code, so the loop is written here, as for
Clef-Flash. It continues training the released LoRA adapter on its Gemma 4 base (the adapter is
loaded trainable; no new LoRA). Each question is turned into tokens by chon's own
``DecisionModel.encode`` with the request's layout, state fitting and option order, exactly as
``chon.serve`` reads it, and the loss is cross-entropy over the option letters' logits at the reply
position (``chon.modeling.letter_logits``), the same numbers serving turns into probabilities. The
result is saved as a release package (adapter + ``chon_config.json`` naming the base model), so
``chon.serve --model <checkpoint>`` loads it like the released one. The saved config drops the
release's fitted temperatures, post-calibration and thinking: they were fitted to the released
weights, and the fine-tuned model is read raw (temperature 1) before our shared calibration, as Kev is.

Serving defaults keep the comparison with Kev fair: one forward pass per question
(``--no-think``, ``--reread off``), no request deadline, and no yes/no commit band (it
would snap probabilities to 0.2/0.8). The model config's ``serve_args`` replaces them.
Chon still applies its own per-type temperatures from ``chon_config.json``; our
calibration fits one more temperature on top, as for every backend.

A serving run (``serving``) reads the option letters through vLLM (``--reader vllm``: the adapter merged into its
base, bf16, prefix caching), the engine Gemma is served with. vLLM lives in its own venv (installed by
sagemaker/entry.py); it is started through a wrapper that keeps the image's Python paths out of it, and
``CHON_VLLM_STRICT=1`` stops the run if it does not start rather than timing the in-process reader.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .base import Backend, Prediction, strip_labels
from .kev import _free_port, answers_to_log_probs

DEFAULT_SERVE_ARGS = ["--no-think", "--reread", "off", "--deadline", "0", "--noul-commit", "off"]


# Settings of the released chon_config.json that were fitted to the released weights or change how an answer is
# read beyond one pass; a fine-tuned package drops them (temperature 1 everywhere, one plain read).
_FITTED_KEYS = ("temperatures", "temperature_categories", "post_temperatures", "think", "pair", "reread",
                "noul_commit", "calibration")


def gold_label(question: dict[str, Any]) -> str:
    """The gold option label of a labelled question, as chon names options (yes/no, levels as "0".."n-1")."""
    label = question["label"]
    if question["type"] == "noul":
        return "yes" if label else "no"
    if question["type"] == "score":
        return str(int(label))
    return str(label)


def training_examples(dm: Any, records: list[dict[str, Any]]) -> list[tuple[list[int], int, int]]:
    """(token ids, number of options, gold option index) per question, encoded exactly as ``DecisionModel._predict``
    reads a request: the state fitted once, the request's layout and header list, the natural option order."""
    examples = []
    for record in records:
        state = record["state"] if record["state"] is not None else ""
        names = list(record["questions"])
        questions = {n: {k: v for k, v in record["questions"][n].items() if k != "label"} for n in names}
        fitted, _ = dm.builder.fit_state(state)
        asked_all = dm.builder.asked_list([questions[n] for n in names])
        layout = dm.layout_for(len(asked_all) if dm.layout == "hybrid_header" else len(names))
        asked = asked_all if layout == "header" else None
        for n in names:
            order = dm.builder.natural_order(questions[n])
            if len(order) > len(dm.builder.letter_ids):
                raise ValueError(f"{record['id']}/{n}: {len(order)} options; Chon reads at most {len(dm.builder.letter_ids)}")
            gold = gold_label(record["questions"][n])
            if gold not in order:
                raise ValueError(f"{record['id']}/{n}: gold label {gold!r} is not one of the options {order}")
            ids = dm.encode(state, questions[n], order, fitted_state=fitted, layout=layout, asked=asked)
            examples.append((ids, len(order), order.index(gold)))
    return examples


def finetuned_config(release_cfg: dict[str, Any], train_info: dict[str, Any]) -> dict[str, Any]:
    cfg = {k: v for k, v in release_cfg.items() if k not in _FITTED_KEYS}
    cfg["temperatures"] = {"choice": 1.0, "noul": 1.0, "score": 1.0}
    cfg["fine_tuned"] = train_info
    return cfg


def clean_vllm_bin(venv: Path | None = None, directory: Path | None = None) -> Path:
    """A ``vllm`` launcher that drops the image's Python paths, then runs the vLLM venv's own binary.

    chon.serve starts the vLLM reader with its own environment, which the training image fills with
    PYTHONPATH and friends; vLLM brings its own torch and must not load the image's."""
    from .prompt_llm import DEFAULT_VLLM_VENV

    real = Path(venv or DEFAULT_VLLM_VENV) / "bin" / "vllm"
    directory = Path(directory or tempfile.mkdtemp(prefix="chon-vllm-"))
    wrapper = directory / "vllm"
    wrapper.write_text(f"#!/bin/sh\nunset PYTHONPATH PYTHONHOME VIRTUAL_ENV CONDA_PREFIX\nexec {real} \"$@\"\n")
    wrapper.chmod(0o755)
    return wrapper


class ChonBackend(Backend):
    concurrent_predictor = True   # predictor forwards to chon.serve, which handles concurrent requests itself

    def train(self, records, out_dir: Path, seed: int, replay: int = 0, replay_records=None) -> Path:
        """Continue training the released adapter; returns a release-style package (see the module docstring)."""
        if replay:
            raise NotImplementedError("Chon has no general-purpose replay data; use groups with replay: 0")
        import contextlib
        import random
        import shutil

        import torch
        from peft import PeftModel
        from chon.infer import DecisionModel, resolve_package
        from chon.modeling import CONFIG_NAME, left_pad, letter_logits, load_causal_lm, load_tokenizer, read_model_config, write_model_config

        cfg = self.train_config
        device = os.environ.get("CHON_TRAIN_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        random.seed(seed)
        torch.manual_seed(seed)

        package = resolve_package(self.config["checkpoint"])
        release_cfg = read_model_config(package)
        base = release_cfg.get("base_model") or self.config["base"]
        read_cfg = {k: v for k, v in release_cfg.items() if k not in ("think", "pair", "reread")}
        read_cfg["prefix_cache"] = False   # training reads every question in its own sequence
        tok_dir = package if os.path.exists(os.path.join(package, "tokenizer_config.json")) else base
        tokenizer = load_tokenizer(tok_dir)
        model = load_causal_lm(base, dtype=cfg.get("weights_dtype", "bfloat16"), device_map=device)
        # the adapter's own weights train in fp32 (PEFT upcasts them by default), the frozen base stays bf16
        model = PeftModel.from_pretrained(model, package, is_trainable=True)
        dm = DecisionModel(model, tokenizer, read_cfg, device=device)   # prompt builder, layout, letter ids, uncapped logits
        if cfg.get("checkpointing", 1):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        model.config.use_cache = False
        model.train()

        examples = training_examples(dm, records)
        n_tokens = [len(ids) for ids, _, _ in examples]
        print(f"chon train: {len(records)} records, {len(examples)} questions, tokens max {max(n_tokens)} "
              f"mean {sum(n_tokens) / len(n_tokens):.0f}", flush=True)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("the released adapter loaded with no trainable parameters")
        print(f"chon train: {sum(p.numel() for p in params):,} trainable parameters ({params[0].dtype})", flush=True)
        batch, accum, epochs = int(cfg.get("batch", 2)), int(cfg.get("accum", 4)), int(cfg.get("epochs", 1))
        steps_per_epoch = -(-len(examples) // (batch * accum))
        total = steps_per_epoch * epochs
        warmup = int(total * float(cfg.get("warmup_ratio", 0.05)))
        optimizer = torch.optim.AdamW(params, lr=float(cfg.get("lr", 2e-5)), weight_decay=float(cfg.get("weight_decay", 0.0)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: (s + 1) / max(1, warmup) if s < warmup else max(0.0, (total - s) / max(1, total - warmup)))
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        letters = torch.as_tensor(dm.builder.letter_ids, dtype=torch.long, device=device)
        use_bf16 = cfg.get("dtype", "bf16") == "bf16" and device.startswith("cuda")
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
        step, t0 = 0, time.time()
        for epoch in range(epochs):
            order = list(range(len(examples)))
            random.shuffle(order)
            micro = [order[i:i + batch] for i in range(0, len(order), batch)]
            running = []
            for mi, idx in enumerate(micro):
                seqs = [examples[i][0] for i in idx]
                ids, mask = left_pad(seqs, pad_id)
                ids, mask = ids.to(device), mask.to(device)
                with autocast:
                    logits = letter_logits(model, ids, mask, letters)   # [batch, 16] float32
                n_opts = torch.tensor([examples[i][1] for i in idx], device=device)
                invalid = torch.arange(logits.shape[1], device=device)[None, :] >= n_opts[:, None]
                logits = logits.masked_fill(invalid, float("-inf"))   # only the question's own letters compete
                gold = torch.tensor([examples[i][2] for i in idx], device=device)
                loss = torch.nn.functional.cross_entropy(logits, gold)
                (loss * len(idx) / (batch * accum)).backward()
                running.append(loss.item())
                if (mi + 1) % accum == 0 or mi == len(micro) - 1:
                    torch.nn.utils.clip_grad_norm_(params, float(cfg.get("max_grad_norm", 1.0)))
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if step % 20 == 0 or step == total:
                        print(f"chon train: epoch {epoch + 1} step {step}/{total} loss {sum(running) / len(running):.4f} "
                              f"lr {scheduler.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
                        running = []

        checkpoint = out_dir / "checkpoint"
        model.save_pretrained(checkpoint)
        for name in os.listdir(package):   # tokenizer and chat template files, as in the release package
            if name.startswith(("tokenizer", "special_tokens", "chat_template")) or name in ("processor_config.json", "preprocessor_config.json", "generation_config.json"):
                shutil.copy2(os.path.join(package, name), checkpoint / name)
        info = {"from": self.config["checkpoint"], "seed": seed, "records": len(records), "questions": len(examples),
                "epochs": epochs, "lr": float(cfg.get("lr", 2e-5)), "batch": batch, "accum": accum, "steps": step}
        write_model_config(str(checkpoint), finetuned_config(release_cfg, info))
        assert (checkpoint / CONFIG_NAME).exists()
        return checkpoint

    def predict(self, checkpoint: Path | None, records: list[dict[str, Any]]) -> list[Prediction]:
        with self.predictor(checkpoint) as predict_one:
            return [predict_one(record) for record in records]

    def inference_precision(self) -> str:
        dtype = str(self.config.get("dtype", "bfloat16"))
        return f"{dtype} (chon --reader vllm)" if self.serving else f"{dtype} (chon, transformers reader)"

    def serve_args(self) -> list[str]:
        args = self.serving_config.get("serve_args") or self.config.get("serve_args", DEFAULT_SERVE_ARGS)
        return [str(a) for a in args]

    def serve_env(self, vllm_bin: Path | None = None) -> dict[str, str]:
        """Environment for ``chon.serve``: no reread or deadline from the host; with the vLLM reader, where its
        binary is, where the merged checkpoint goes, and stop rather than fall back to the in-process reader."""
        env = dict(os.environ)
        env.pop("CHON_REREAD", None)
        env.pop("CHON_DEADLINE_S", None)
        if self.serving:
            env["CHON_VLLM_STRICT"] = "1"
            # decision_model.backends.chon_serve (serve_command) runs in the child: put this package on its path
            src = str(Path(__file__).resolve().parents[2])
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
            env.setdefault("CHON_MERGED_DIR", "/opt/ml/models/chon-merged")   # on the job's data volume
            if vllm_bin is not None:
                env["CHON_VLLM_BIN"] = str(vllm_bin)
        return env

    def serve_command(self, port: int, checkpoint: Path | None = None) -> list[str]:
        args = self.serve_args()
        model = str(checkpoint) if checkpoint is not None else self.config["checkpoint"]
        # Serving profile: chon.serve without its one-request-at-a-time lock, so its vLLM reader batches concurrent
        # requests like every other model's server (decision_model.backends.chon_serve). One request runs as before.
        module = "decision_model.backends.chon_serve" if self.serving else "chon.serve"
        return [sys.executable, "-m", module, "--model", model,
                "--dtype", str(self.config.get("dtype", "bfloat16")), "--port", str(port), *args]

    @contextmanager
    def predictor(self, checkpoint: Path | None) -> Iterator[Callable[[dict[str, Any]], Prediction]]:
        """Start ``chon.serve`` once (the release, or a fine-tuned package from ``train``); each call sends one
        record to ``/v1/systemone``."""
        port = _free_port()
        env = self.serve_env(clean_vllm_bin() if self.serving else None)
        server = subprocess.Popen(self.serve_command(port, checkpoint), env=env)
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
