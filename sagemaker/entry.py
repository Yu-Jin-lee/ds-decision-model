#!/usr/bin/env python3
"""SageMaker training-job entry point.

Installs what the chosen backend needs, then runs scripts/run.py with the
job's hyperparameters (--model, --group, --seed, optional --final).
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
    sh(sys.executable, str(REPO / "scripts" / "run.py"), *args)
    upload_outputs()


def upload_outputs() -> None:
    """Copy the run folder to SM_OUTPUT_PREFIX/{run_id}/ so results are readable
    without unpacking model.tar.gz, and so the launcher can skip finished runs."""
    prefix = os.environ.get("SM_OUTPUT_PREFIX")
    model_dir = Path(os.environ.get("SM_MODEL_DIR", "runs"))
    if not prefix:
        return
    import boto3

    bucket, _, key_prefix = prefix.removeprefix("s3://").partition("/")
    client = boto3.client("s3")
    # summary.json last, so its presence means the upload finished.
    files = sorted((p for p in model_dir.rglob("*") if p.is_file()), key=lambda p: p.name == "summary.json")
    for path in files:
        key = f"{key_prefix.rstrip('/')}/{path.relative_to(model_dir).as_posix()}".lstrip("/")
        client.upload_file(str(path), bucket, key)
    print(f"uploaded {len(files)} files to {prefix}", flush=True)


if __name__ == "__main__":
    main()
