#!/usr/bin/env python3
"""SageMaker training-job entry point.

Installs what the chosen backend needs, then runs scripts/run.py with the
job's hyperparameters (--model, --group, --seed, optional --final).

Serving jobs (launcher --serving) re-time finished runs with the serving profile
(configs/serving.yaml): the job installs each model's serving engine as well, downloads
the checkpoint of every trained run from the runs prefix, and writes serving/{run_id}/.

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
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def sh(*command: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True, cwd=cwd, env=env)


def main() -> None:
    args, serving = take_serving_flag(sys.argv[1:])
    model = args[args.index("--model") + 1]
    from decision_model.config import load_model_config, load_yaml, CONFIG_DIR

    config = load_model_config(model)
    serving_config = load_yaml(CONFIG_DIR / "serving.yaml") if serving else {}
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
        if serving:
            # e.g. flash-linear-attention for Kev's fused kernels (not in Kev's serve extra)
            for package in (config.get("serving") or {}).get("install", []):
                sh("uv", "pip", "install", "--python", str(kev_dir / ".venv" / "bin" / "python"), package,
                   cwd=kev_dir, env=uv_environment(kev_dir))
    elif config["backend"] == "chon":
        sh(sys.executable, "-m", "pip", "install", "-q", f"chon[serve] @ git+{config['chon_repo']}@{config['chon_ref']}")
        if serving:
            install_vllm(serving_config["vllm_package"])   # Chon's --reader vllm
    elif config["backend"] == "prompt_llm":
        from decision_model.backends.prompt_llm import _clean_env

        package = serving_config["vllm_package"] if serving else config.get("vllm", {}).get("package", "vllm")
        venv_python = install_vllm(package)
        env = _clean_env()
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
    if serving and "--resume-from" in args:
        raise SystemExit("--resume-from does not apply to a serving job; it reads checkpoints from the runs prefix")
    if "--resume-from" in args:
        resume_root = fetch_resume(args[args.index("--resume-from") + 1])
        args = _drop(args, "--resume-from")
    if "--runs" in args:
        run_batch(model, parse_runs(args[args.index("--runs") + 1]), _without(args, "--runs"), resume_root, serving)
        return
    group, seed = args[args.index("--group") + 1], args[args.index("--seed") + 1] if "--seed" in args else "0"
    if serving:
        args = [*args, "--serving", *serving_checkpoint_args(model, group, int(seed))]
    elif resume_root is not None:
        args = [*args, *resume_args(resume_root, f"{model}/{group}/s{seed}")]
    sh(sys.executable, str(REPO / "scripts" / "run.py"), *args)
    upload_outputs()


def take_serving_flag(args: list[str]) -> tuple[list[str], bool]:
    """SageMaker passes the launcher's ``serving`` hyperparameter as ``--serving 1``; run.py takes a bare flag."""
    if "--serving" not in args:
        return args, False
    i = args.index("--serving")
    has_value = i + 1 < len(args) and not args[i + 1].startswith("--")
    if has_value and args[i + 1].lower() in ("0", "false", "no"):
        return args[:i] + args[i + 2:], False
    return args[:i] + args[i + (2 if has_value else 1):], True


def install_vllm(package: str) -> str:
    """vLLM brings its own torch/CUDA wheels: install it in a separate venv, never the image's Python."""
    from decision_model.backends.prompt_llm import DEFAULT_VLLM_VENV, _clean_env

    sh(sys.executable, "-m", "pip", "install", "-q", "uv")
    env = _clean_env()
    sh("uv", "venv", "--python", "3.12", str(DEFAULT_VLLM_VENV), env=env)
    venv_python = str(DEFAULT_VLLM_VENV / "bin" / "python")
    sh("uv", "pip", "install", "--python", venv_python, package, env=env)
    return venv_python


def checkpoint_relative_path(summary: dict, run_id: str) -> str:
    """Where a run's checkpoint sits inside its run folder (``checkpoint/checkpoint`` for Kev and Chon,
    ``checkpoint`` for Laya), read from the path its summary.json recorded."""
    recorded = str(summary.get("checkpoint") or "")
    marker = f"/{run_id}/"
    if marker not in recorded:
        raise ValueError(f"{run_id}: summary.json records checkpoint {recorded!r}, not inside the run folder")
    return recorded.split(marker, 1)[1].strip("/")


def serving_checkpoint_args(model: str, group: str, seed: int) -> list[str]:
    """``--checkpoint DIR`` for a trained run: its checkpoint downloaded from the runs prefix (SM_OUTPUT_PREFIX);
    nothing for a run that does not train (B0, P0: the released model)."""
    from decision_model.pipeline import find_run

    run = find_run(model, group, seed)
    if not run.train:
        return []
    prefix = os.environ.get("SM_OUTPUT_PREFIX")
    if not prefix:
        raise SystemExit(f"{run.run_id}: a serving job needs SM_OUTPUT_PREFIX to find the run's checkpoint")
    import json

    import boto3

    client = boto3.client("s3")
    bucket, _, key_prefix = prefix.removeprefix("s3://").partition("/")
    run_key = f"{key_prefix.rstrip('/')}/{run.run_id}".lstrip("/")
    summary = json.loads(client.get_object(Bucket=bucket, Key=f"{run_key}/summary.json")["Body"].read())
    relative = checkpoint_relative_path(summary, run.run_id)
    # Outside SM_MODEL_DIR, so the checkpoint is not uploaded again with the serving run.
    local = Path(os.environ.get("SERVING_CHECKPOINTS_DIR", "/opt/ml/serving-checkpoints")) / run.run_id / relative
    count = 0
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{run_key}/{relative}/"):
        for item in page.get("Contents", []):
            target = local / item["Key"][len(f"{run_key}/{relative}/"):]
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, item["Key"], str(target))
            count += 1
    if not count:
        raise SystemExit(f"{run.run_id}: no checkpoint files under s3://{bucket}/{run_key}/{relative}/")
    print(f"serving {run.run_id}: {count} checkpoint files -> {local}", flush=True)
    return ["--checkpoint", str(local)]


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


def run_batch(model: str, runs: list[tuple[str, int]], extra: list[str], resume_root: Path | None = None,
              serving: bool = False) -> None:
    import shutil

    model_dir = Path(os.environ.get("SM_MODEL_DIR", "runs"))
    failed = []
    for i, (group, seed) in enumerate(runs, 1):
        run_id = f"{model}/{group}/s{seed}"
        if serving:
            run_id = f"serving/{run_id}"   # pipeline.SERVING_DIR
        print(f"=== [{i}/{len(runs)}] {run_id} ({disk_free()})", flush=True)
        before = _merged_entries()
        more: list[str] = []
        try:
            more = (["--serving", *serving_checkpoint_args(model, group, seed)] if serving
                    else resume_args(resume_root, run_id))
            sh(sys.executable, str(REPO / "scripts" / "run.py"), "--model", model, "--group", group, "--seed", str(seed),
               *extra, *more)
        except (subprocess.CalledProcessError, SystemExit, OSError) as error:
            code = (error.returncode if isinstance(error, subprocess.CalledProcessError)
                    else error.code if isinstance(error, SystemExit) else repr(error))
            print(f"!!! {run_id} failed (exit {code}); continuing", flush=True)
            failed.append(run_id)
            continue
        finally:
            if serving:
                free_serving_disk(more, before)
        upload_outputs(run_id)
        # Results are in S3 now; free the disk and keep model.tar.gz from duplicating them.
        shutil.rmtree(model_dir / run_id, ignore_errors=True)
    print(f"=== batch done: {len(runs) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)
    if failed:
        sys.exit(1)


def disk_free(path: str = "/opt/ml") -> str:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return "disk ?"
    return f"{usage.free / 2**30:.0f} GB free of {usage.total / 2**30:.0f} GB on {path}"


def _merged_dir() -> Path:
    return Path(os.environ.get("CHON_MERGED_DIR", "/opt/ml/models/chon-merged"))


def _merged_entries() -> set[Path]:
    directory = _merged_dir()
    return set(directory.iterdir()) if directory.is_dir() else set()


def free_serving_disk(more: list[str], merged_before: set[Path]) -> None:
    """After a serving run of a trained checkpoint: delete its downloaded checkpoint and the merged copy the
    Chon vLLM reader made of it (one full model per adapter), so a batch of trained runs does not fill the disk.
    Merged copies of the released adapter (B0) stay, since every B0 run reuses them."""
    if "--checkpoint" not in more:
        return
    root = Path(os.environ.get("SERVING_CHECKPOINTS_DIR", "/opt/ml/serving-checkpoints"))
    shutil.rmtree(root, ignore_errors=True)
    for entry in _merged_entries() - merged_before:
        shutil.rmtree(entry, ignore_errors=True) if entry.is_dir() else entry.unlink(missing_ok=True)


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
