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
    elif config["backend"] == "prompt_llm":
        # vLLM brings its own torch/CUDA wheels: install it in a separate venv, never the image's Python.
        from decision_model.backends.prompt_llm import DEFAULT_VLLM_VENV, _clean_env

        sh(sys.executable, "-m", "pip", "install", "-q", "uv")
        env = _clean_env()
        sh("uv", "venv", "--python", "3.12", str(DEFAULT_VLLM_VENV), env=env)
        venv_python = str(DEFAULT_VLLM_VENV / "bin" / "python")
        sh("uv", "pip", "install", "--python", venv_python, config.get("vllm", {}).get("package", "vllm"), env=env)
        # Download once, then split into small shards: vLLM mmaps each .safetensors file whole,
        # and Gemma's 50 GB shard cannot be mapped with the instance's 32 GB of RAM.
        models = Path(os.environ.get("VLLM_MODELS_DIR", "/opt/ml/models"))
        raw, local = models / "download", models / config["name"]
        sh("df", "-h")
        sh(venv_python, "-c", "import sys; from huggingface_hub import snapshot_download; "
           "snapshot_download(sys.argv[1], local_dir=sys.argv[2])", config["checkpoint"], str(raw), env=env)
        sh(sys.executable, str(REPO / "scripts" / "reshard_safetensors.py"), str(raw), str(local),
           "--max-gb", str(config.get("vllm", {}).get("shard_gb", 4)), "--consume")
        sh("df", "-h")
        import shutil

        shutil.rmtree(raw)
        os.environ["VLLM_MODEL_PATH"] = str(local)   # read by the backend in every run.py process
    resume_root = None
    if "--resume-from" in args:
        resume_root = fetch_resume(args[args.index("--resume-from") + 1])
        args = _drop(args, "--resume-from")
    if "--runs" in args:
        run_batch(model, parse_runs(args[args.index("--runs") + 1]), _without(args, "--runs"), resume_root)
        return
    if resume_root is not None:
        group, seed = args[args.index("--group") + 1], args[args.index("--seed") + 1] if "--seed" in args else "0"
        args = [*args, *resume_args(resume_root, f"{model}/{group}/s{seed}")]
    sh(sys.executable, str(REPO / "scripts" / "run.py"), *args)
    upload_outputs()


def _drop(args: list[str], flag: str) -> list[str]:
    """``args`` without ``flag`` and its value."""
    i = args.index(flag)
    return args[:i] + args[i + 2:]


def fetch_resume(uri: str) -> Path:
    """Download and unpack an earlier job's model.tar.gz (s3://... or a local path).

    It holds that job's run folders ({model}/{group}/s{seed}/), including the
    checkpoints of runs that trained but failed afterwards."""
    import tarfile

    root = Path(os.environ.get("RESUME_DIR", "/opt/ml/resume"))
    root.mkdir(parents=True, exist_ok=True)
    archive = Path(uri)
    if uri.startswith("s3://"):
        import boto3

        bucket, _, key = uri.removeprefix("s3://").partition("/")
        archive = root / "model.tar.gz"
        print(f"+ download {uri}", flush=True)
        boto3.client("s3").download_file(bucket, key, str(archive))
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    if uri.startswith("s3://"):
        archive.unlink()
    return root


def resume_args(resume_root: Path | None, run_id: str) -> list[str]:
    """``--checkpoint DIR`` if the earlier job left a checkpoint for this run, else nothing (train as usual)."""
    if resume_root is None:
        return []
    checkpoint = resume_root / run_id / "checkpoint" / "checkpoint"
    if checkpoint.is_dir():
        print(f"resume {run_id} from {checkpoint}", flush=True)
        return ["--checkpoint", str(checkpoint)]
    print(f"resume {run_id}: no checkpoint in the earlier job, training from the start", flush=True)
    return []


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


def run_batch(model: str, runs: list[tuple[str, int]], extra: list[str], resume_root: Path | None = None) -> None:
    import shutil

    model_dir = Path(os.environ.get("SM_MODEL_DIR", "runs"))
    failed = []
    for i, (group, seed) in enumerate(runs, 1):
        run_id = f"{model}/{group}/s{seed}"
        print(f"=== [{i}/{len(runs)}] {run_id}", flush=True)
        try:
            sh(sys.executable, str(REPO / "scripts" / "run.py"), "--model", model, "--group", group, "--seed", str(seed),
               *extra, *resume_args(resume_root, run_id))
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
