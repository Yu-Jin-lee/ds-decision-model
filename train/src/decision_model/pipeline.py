"""One run end to end: train -> fit temperature -> predict -> score -> save."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .backends import Backend, get_backend
from .config import CONFIG_DIR, REPO_ROOT, Run, expand_runs, load_model_config, load_yaml
from .data import build_records, data_manifest, read_jsonl, write_jsonl
from .evaluation import fit_model_temperature, score
from .formats import load_task_specs
from .latency import measure as measure_latency
from .latency import measure_by_view


# Serving runs (configs/serving.yaml) are written under runs/serving/{model}/{group}/s{seed}/, next to the
# evaluation run they re-time, so they never overwrite it.
SERVING_DIR = "serving"


def find_run(model: str, group: str, seed: int, config_dir: Path = CONFIG_DIR) -> Run:
    groups_config = load_yaml(config_dir / "groups.yaml")
    tasks_config = load_yaml(config_dir / "tasks.yaml")
    base_group = _base_group(group, groups_config)
    runs = expand_runs(model, groups_config, tasks_config, groups={base_group}, include_optional=True)
    for run in runs:
        if run.group == group and run.seed == seed:
            return run
    available = sorted({r.group for r in runs})
    raise ValueError(f"no run {group!r} seed {seed}; groups from {base_group!r}: {available}")


def _base_group(group: str, groups_config: dict[str, Any]) -> str:
    """Map an expanded name (e.g. ``S-s4-noul``) back to its config group."""
    names = sorted(groups_config["groups"], key=len, reverse=True)
    for name in names:
        if group == name or group.startswith(name + "-"):
            return name
    raise ValueError(f"unknown group {group!r}")


def _git_commit() -> str | None:
    """Commit of the code that ran. On SageMaker the launcher records it at submit
    time (DECISION_MODEL_GIT_COMMIT), because git refuses to read the copied repo."""
    if os.environ.get("DECISION_MODEL_GIT_COMMIT"):
        return os.environ["DECISION_MODEL_GIT_COMMIT"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    """True if the code had uncommitted changes when it was run or submitted."""
    if "DECISION_MODEL_GIT_DIRTY" in os.environ:
        return os.environ["DECISION_MODEL_GIT_DIRTY"] == "1"
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", "."], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(status.strip())


def save_model_inputs(out_dir: Path, partition: str, records: list[dict[str, Any]]) -> str:
    """Write the exact records the model saw (assembled state, questions, gold) as
    ``model_inputs_{partition}.jsonl.gz`` and return their sha256.

    The data in S3 keeps raw fields and the state is assembled at load time, so
    this file is the record of what a run actually trained and was scored on.
    """
    lines = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records).encode("utf-8")
    with open(out_dir / f"model_inputs_{partition}.jsonl.gz", "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(lines)
    return hashlib.sha256(lines).hexdigest()


def execute(
    run: Run,
    data_root: Path,
    out_root: Path,
    final: bool = False,
    replay_file: Path | None = None,
    backend: Backend | None = None,
    config_dir: Path = CONFIG_DIR,
    measure_latency_too: bool = True,
    checkpoint: Path | None = None,
    serving: bool = False,
) -> dict[str, Any]:
    """``checkpoint``: a checkpoint this run already trained (e.g. from a failed job's
    model.tar.gz). Training is skipped and everything after it runs as usual.

    ``serving``: re-time the run with the model's serving profile (configs/serving.yaml): bf16 and its own
    fastest engine. Nothing is trained; a trained run needs its ``checkpoint``, which is read where it is.
    Calibration, development predictions, metrics and latency are produced as usual, so the summary also
    shows whether the serving path answers like the evaluation path."""
    model_config = load_model_config(run.model, config_dir)
    tasks_config = load_yaml(config_dir / "tasks.yaml")
    specs = load_task_specs(tasks_config)
    input_hashes: dict[str, str] = {}
    backend = backend or get_backend(model_config, serving=serving)
    out_dir = Path(out_root) / (SERVING_DIR if serving else "") / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    resumed_from = None
    if serving:
        if run.train and checkpoint is None:
            raise ValueError(f"{run.run_id} is a trained run: a serving run needs its checkpoint (--checkpoint)")
        if checkpoint is not None and not Path(checkpoint).exists():
            raise FileNotFoundError(f"checkpoint {checkpoint} does not exist")
        resumed_from = str(checkpoint) if checkpoint is not None else None
    elif checkpoint is not None:
        if not run.train:
            raise ValueError(f"{run.run_id} does not train; it has no checkpoint to resume from")
        if not Path(checkpoint).exists():
            raise FileNotFoundError(f"checkpoint {checkpoint} does not exist")
        resumed_from = str(checkpoint)
        # Keep the run folder's usual layout (checkpoint/checkpoint), so it is uploaded with the run.
        local = out_dir / "checkpoint" / "checkpoint"
        if Path(checkpoint).resolve() != local.resolve():
            shutil.copytree(checkpoint, local, dirs_exist_ok=True)
        checkpoint = local
    replay_records = None
    if run.train and not serving:
        train_records = build_records(data_root, specs, run.levels, run.train_langs, "train", with_labels=True)
        input_hashes["train"] = save_model_inputs(out_dir, "train", train_records)
    if run.train and checkpoint is None and not serving:
        if run.replay and model_config["backend"] != "kev":
            if replay_file is None or not Path(replay_file).exists():
                raise FileNotFoundError(
                    f"{run.run_id} uses replay={run.replay}; pass --replay-file (Kev-format JSONL) for this backend"
                )
            replay_records = read_jsonl(replay_file)
        checkpoint = backend.train(train_records, out_dir / "checkpoint", run.seed, run.replay, replay_records)

    if getattr(backend, "generative", False):
        # Parsed text answers carry no probabilities: nothing to calibrate.
        temperature = 1.0
    else:
        calibration = build_records(data_root, specs, run.levels, run.eval_langs, "calibration", with_labels=True, for_prediction=True)
        input_hashes["calibration"] = save_model_inputs(out_dir, "calibration", calibration)
        temperature = fit_model_temperature(calibration, backend.predict(checkpoint, calibration))

    summary: dict[str, Any] = {
        "run": run.to_dict(),
        "model_config": model_config,
        "temperature": temperature,
        "checkpoint": str(checkpoint) if checkpoint else model_config["checkpoint"],
        "resumed_from": resumed_from,
        "profile": "serving" if serving else "evaluation",
        "serving": load_yaml(config_dir / "serving.yaml") if serving else None,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "data_manifest": data_manifest(data_root),
        # How the raw data became model input, and a hash of exactly what the model saw.
        "task_configs": {task: tasks_config["tasks"][task] for task in run.levels},
        "model_inputs_sha256": input_hashes,
    }
    partitions = ["development"] + (["test"] if final else [])
    for partition in partitions:
        records = build_records(data_root, specs, run.levels, run.eval_langs, partition, with_labels=True, for_prediction=True)
        input_hashes[partition] = save_model_inputs(out_dir, partition, records)
        predictions = backend.predict(checkpoint, records)
        rows, metrics = score(specs, records, predictions, temperature)
        write_jsonl(out_dir / f"predictions_{partition}.jsonl", predictions)
        write_jsonl(out_dir / f"scored_{partition}.jsonl", rows)
        summary[f"metrics_{partition}"] = metrics
        if partition == "development" and measure_latency_too:
            # Same conditions for every model; see latency.py.
            with backend.predictor(checkpoint) as predict_one:
                latency = measure_latency(predict_one, records)
                # Levels that train several formats (e.g. both) can be served with one
                # question; time each format alone on the same model and records.
                by_view = measure_by_view(predict_one, records, specs, run.levels)
            latency.setdefault("conditions", {})["precision"] = backend.inference_precision()
            summary["latency_development"] = latency
            if by_view:
                summary["latency_development_by_view"] = by_view

    summary["seconds"] = round(time.time() - started, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
