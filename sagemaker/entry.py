#!/usr/bin/env python3
"""SageMaker training-job entry point.

Installs what the chosen backend needs, then runs scripts/run.py with the
job's hyperparameters (--model, --group, --seed, optional --final).

Batch jobs (launcher --batch) pass --runs "GROUP:SEED,GROUP:SEED,...": the
installs and the base-model download happen once, then each run executes in its
own run.py process (so every run starts from the base checkpoint with fresh
memory and RNG state). Each run is uploaded as soon as it finishes and its local
folder removed; a failed run does not stop the rest.
For the main runs, bake these installs into one container image instead,
so every job uses the same versions (design doc: one image for all runs).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def sh(*command: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=cwd, env=env)


def main() -> None:
    args = sys.argv[1:]
    model = args[args.index("--model") + 1]
    from decision_model.config import load_model_config

    config = load_model_config(model)
    sh(sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt"))
    if config["backend"] == "clef":
        sh(sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-clef.txt"))
    elif config["backend"] == "laya":
        sh(sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements-laya.txt"))
    elif config["backend"] == "kev":
        kev_dir = Path(os.environ.get("KEV_DIR", "/opt/kev"))
        if not kev_dir.exists():
            sh("git", "clone", "--depth", "1", "--branch", config.get("kev_ref", "main"), config["kev_repo"], str(kev_dir))
        sh(sys.executable, "-m", "pip", "install", "-q", "uv")
        from decision_model.backends.kev import uv_environment

        sh("uv", "sync", "--extra", "serve", cwd=kev_dir, env=uv_environment(kev_dir))
    if "--runs" in args:
        run_batch(model, parse_runs(args[args.index("--runs") + 1]), _without(args, "--runs"))
        return
    sh(sys.executable, str(REPO / "scripts" / "run.py"), *args)
    upload_outputs()


def parse_runs(value: str) -> list[tuple[str, int]]:
    """``"S-s5-choice:0,S-s5-score:1"`` -> ``[("S-s5-choice", 0), ("S-s5-score", 1)]``."""
    runs = []
    for item in value.split(","):
        group, sep, seed = item.strip().rpartition(":")
        if not sep or not group:
            raise ValueError(f"--runs item must be GROUP:SEED, got {item!r}")
        runs.append((group, int(seed)))
    return runs


def _without(args: list[str], *flags: str) -> list[str]:
    """Drop the given flags and their values; keep the rest (e.g. --final)."""
    out, skip = [], False
    for arg in args:
        if skip:
            skip = False
            continue
        if arg in flags or arg in ("--model", "--group", "--seed"):
            skip = True
            continue
        out.append(arg)
    return out


def run_batch(model: str, runs: list[tuple[str, int]], extra: list[str]) -> None:
    import shutil

    model_dir = Path(os.environ.get("SM_MODEL_DIR", "runs"))
    failed = []
    for i, (group, seed) in enumerate(runs, 1):
        run_id = f"{model}/{group}/s{seed}"
        print(f"=== [{i}/{len(runs)}] {run_id}", flush=True)
        try:
            sh(sys.executable, str(REPO / "scripts" / "run.py"), "--model", model, "--group", group, "--seed", str(seed), *extra)
        except subprocess.CalledProcessError as error:
            print(f"!!! {run_id} failed (exit {error.returncode}); continuing", flush=True)
            failed.append(run_id)
            continue
        upload_outputs(run_id)
        # Results are in S3 now; free the disk and keep model.tar.gz from duplicating them.
        shutil.rmtree(model_dir / run_id, ignore_errors=True)
    print(f"=== batch done: {len(runs) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)
    if failed:
        sys.exit(1)


def upload_outputs(run_id: str | None = None) -> None:
    """Copy the run folder to SM_OUTPUT_PREFIX/{run_id}/ so results are readable
    without unpacking model.tar.gz, and so the launcher can skip finished runs.
    With ``run_id``, only that run's folder."""
    prefix = os.environ.get("SM_OUTPUT_PREFIX")
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "runs"))
    if not prefix:
        return
    import boto3

    bucket, _, key_prefix = prefix.removeprefix("s3://").partition("/")
    client = boto3.client("s3")
    root = model_dir / run_id if run_id else model_dir
    # summary.json last, so its presence means the upload finished.
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.name == "summary.json")
    for path in files:
        key = f"{key_prefix.rstrip('/')}/{path.relative_to(model_dir).as_posix()}".lstrip("/")
        client.upload_file(str(path), bucket, key)
    print(f"uploaded {len(files)} files to {prefix}{run_id or ''}", flush=True)


if __name__ == "__main__":
    main()
