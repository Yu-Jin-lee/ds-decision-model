"""One run end to end: train -> fit temperature -> predict -> score -> save."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from .backends import Backend, get_backend
from .config import CONFIG_DIR, REPO_ROOT, Run, expand_runs, load_model_config, load_yaml
from .data import build_records, data_manifest, read_jsonl, write_jsonl
from .evaluation import fit_model_temperature, score
from .formats import load_task_specs


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
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def execute(
    run: Run,
    data_root: Path,
    out_root: Path,
    final: bool = False,
    replay_file: Path | None = None,
    backend: Backend | None = None,
    config_dir: Path = CONFIG_DIR,
) -> dict[str, Any]:
    model_config = load_model_config(run.model, config_dir)
    specs = load_task_specs(load_yaml(config_dir / "tasks.yaml"))
    backend = backend or get_backend(model_config)
    out_dir = Path(out_root) / run.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    checkpoint = None
    replay_records = None
    if run.train:
        train_records = build_records(data_root, specs, run.levels, run.train_langs, "train", with_labels=True)
        if run.replay and model_config["backend"] != "kev":
            if replay_file is None or not Path(replay_file).exists():
                raise FileNotFoundError(
                    f"{run.run_id} uses replay={run.replay}; pass --replay-file (Kev-format JSONL) for this backend"
                )
            replay_records = read_jsonl(replay_file)
        checkpoint = backend.train(train_records, out_dir / "checkpoint", run.seed, run.replay, replay_records)

    calibration = build_records(data_root, specs, run.levels, run.eval_langs, "calibration", with_labels=True)
    temperature = fit_model_temperature(calibration, backend.predict(checkpoint, calibration))

    summary: dict[str, Any] = {
        "run": run.to_dict(),
        "model_config": model_config,
        "temperature": temperature,
        "checkpoint": str(checkpoint) if checkpoint else model_config["checkpoint"],
        "git_commit": _git_commit(),
        "data_manifest": data_manifest(data_root),
    }
    partitions = ["development"] + (["test"] if final else [])
    for partition in partitions:
        records = build_records(data_root, specs, run.levels, run.eval_langs, partition, with_labels=True)
        predictions = backend.predict(checkpoint, records)
        rows, metrics = score(specs, records, predictions, temperature)
        write_jsonl(out_dir / f"predictions_{partition}.jsonl", predictions)
        write_jsonl(out_dir / f"scored_{partition}.jsonl", rows)
        summary[f"metrics_{partition}"] = metrics

    summary["seconds"] = round(time.time() - started, 1)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary
