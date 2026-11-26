#!/usr/bin/env python3
"""Method ``job``: submit many load-test jobs one slot at a time, without anyone watching.

The serving instance (configs/serving.yaml) has a small quota (here 1 on-demand + 1 spot at a time), so jobs
cannot all be submitted at once. This waits until a slot is free (counting every InProgress job on that
instance type, ours or not), submits the next item with bench/job/launch.py --batch, and repeats until every
item is submitted and finished. Results already in --output are skipped by the launcher.

One item = one model and its groups, in one batch job (installs and downloads once):

    python bench/job/queue_jobs.py --items "gemma-4-26b-a4b:P0-s2+P0-s4,kev-4b:B0-s2-choice+B0-s4-both" \
        --data s3://BUCKET/ds-jf-decision-model/data/ --output s3://BUCKET/ds-jf-decision-model/runs/ \
        --role <SageMaker role ARN> --image-uri <training image>

--remote runs this same queue as a small CPU SageMaker job (ml.m5.large) with the role, so the laptop can be
closed; its log is the job's CloudWatch log. The role needs sagemaker:CreateTrainingJob and iam:PassRole
(AmazonSageMakerFullAccess has both).
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

TERMINAL = ("Completed", "Failed", "Stopped")
CONTROLLER_IMAGE = "763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-training:2.5-cpu-py311"


def log(message: str) -> None:
    print(f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}Z {message}", flush=True)


def parse_items(value: str) -> list[tuple[str, list[str]]]:
    """``"m1:G1+G2,m2:G3"`` -> ``[("m1", ["G1", "G2"]), ("m2", ["G3"])]``. Items are separated by commas, not
    semicolons: SageMaker passes hyperparameters through a shell, where ``;`` ends the command."""
    items = []
    for part in filter(None, (p.strip() for p in value.split(","))):
        model, sep, groups = part.partition(":")
        if not sep or not model or not groups:
            raise ValueError(f"item must be model:GROUP+GROUP, got {part!r}")
        items.append((model.strip(), [g.strip() for g in groups.split("+") if g.strip()]))
    return items


def busy_slots(sagemaker, instance_type: str) -> dict[str, int]:
    """InProgress training jobs on ``instance_type``, split into on-demand and spot."""
    busy = {"ondemand": 0, "spot": 0}
    for page in sagemaker.get_paginator("list_training_jobs").paginate(StatusEquals="InProgress"):
        for summary in page["TrainingJobSummaries"]:
            job = sagemaker.describe_training_job(TrainingJobName=summary["TrainingJobName"])
            if job["ResourceConfig"]["InstanceType"] == instance_type:
                busy["spot" if job.get("EnableManagedSpotTraining") else "ondemand"] += 1
    return busy


def launch_command(model: str, groups: list[str], args: argparse.Namespace, spot: bool) -> list[str]:
    command = [sys.executable, str(REPO / "bench" / "job" / "launch.py"), "--model", model, "--seed", str(args.seed),
               "--data", args.data, "--output", args.output, "--role", args.role, "--image-uri", args.image_uri,
               "--max-run-hours", str(args.max_run_hours), "--batch"]
    for group in groups:
        command += ["--group", group]
    if args.label:
        command += ["--label", args.label]
    return command + (["--spot"] if spot else [])


def submitted_job(output: str) -> str | None:
    """The job name from the launcher's ``submit ... -> NAME`` line; "DONE" when every run already has results."""
    for line in output.splitlines():
        match = re.match(r"submit .* -> (\S+)$", line.strip())
        if match:
            return match.group(1)
    if "nothing to run" in output:
        return "DONE"
    return None


def run_queue(args: argparse.Namespace) -> None:
    import boto3

    from decision_model.config import CONFIG_DIR, load_yaml

    os.environ.setdefault("AWS_DEFAULT_REGION", args.region)
    instance_type = load_yaml(CONFIG_DIR / "serving.yaml")["instance"]
    sagemaker = boto3.client("sagemaker", region_name=args.region)
    pending = parse_items(args.items)
    running: dict[str, str] = {}   # job name -> item label
    results: dict[str, str] = {}
    deadline = time.time() + args.max_hours * 3600
    log(f"queue: {len(pending)} items on {instance_type}, slots on-demand {args.ondemand_slots} spot {args.spot_slots}")
    while (pending or running) and time.time() < deadline:
        try:
            for job, label in list(running.items()):
                described = sagemaker.describe_training_job(TrainingJobName=job)
                if described["TrainingJobStatus"] in TERMINAL:
                    reason = described.get("FailureReason", "")[:300]
                    log(f"{label}: {job} {described['TrainingJobStatus']} {reason}")
                    results[label] = described["TrainingJobStatus"]
                    del running[job]
            if pending:
                busy = busy_slots(sagemaker, instance_type)
                for slot, limit in (("ondemand", args.ondemand_slots), ("spot", args.spot_slots)):
                    if not pending or busy[slot] >= limit:
                        continue
                    model, groups = pending[0]
                    label = f"{model}:{'+'.join(groups)}"
                    out = subprocess.run(launch_command(model, groups, args, spot=slot == "spot"), cwd=REPO,
                                         capture_output=True, text=True,
                                         env={**os.environ, "SAGEMAKER_SUPPRESS_V2_WARNING": "1",
                                              "AWS_DEFAULT_REGION": args.region})   # none set in a job
                    job = submitted_job(out.stdout + out.stderr)
                    if job == "DONE":
                        log(f"{label}: every run already has results; skipped")
                        results[label] = "skipped"
                        pending.pop(0)
                    elif job:
                        log(f"{label}: submitted {job} ({slot})")
                        running[job] = label
                        busy[slot] += 1
                        pending.pop(0)
                    else:
                        tail = (out.stdout + out.stderr).strip().splitlines()[-4:] or [f"exit {out.returncode}"]
                        log(f"{label}: not submitted on {slot} yet: {' | '.join(tail)[:800]}")
        except Exception as error:  # noqa: BLE001 - throttling or a lapsed session: try again next round
            log(f"error: {type(error).__name__}: {error}")
        time.sleep(args.poll_seconds)
    log(f"queue done: {results}; never submitted: {[m for m, _ in pending]}; still running: {list(running)}")


def submit_remote(args: argparse.Namespace) -> None:
    """Run this queue as a CPU SageMaker job, so nothing has to stay on the laptop."""
    import importlib.util

    from sagemaker.estimator import Estimator

    spec = importlib.util.spec_from_file_location("launch_sagemaker", REPO / "scripts" / "launch_sagemaker.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    commit, dirty = launcher.git_state()
    hyperparameters = {"items": args.items, "data": args.data, "output": args.output, "role": args.role,
                       "image-uri": args.image_uri, "seed": args.seed, "max-run-hours": args.max_run_hours,
                       "region": args.region, "ondemand-slots": args.ondemand_slots, "spot-slots": args.spot_slots,
                       "max-hours": args.max_hours, "poll-seconds": args.poll_seconds,
                       **({"label": args.label} if args.label else {})}
    estimator = Estimator(
        image_uri=CONTROLLER_IMAGE.format(region=args.region), role=args.role, instance_type="ml.m5.large",
        instance_count=1, volume_size=30, entry_point="bench/job/queue_jobs.py", source_dir=str(REPO),
        hyperparameters=hyperparameters, output_path=f"{args.output.rstrip('/')}/_jobs/",
        code_location=f"{args.output.rstrip('/')}/_code", base_job_name="bench-queue",
        max_run=int((args.max_hours + 1) * 3600),
        environment={"DECISION_MODEL_GIT_COMMIT": commit, "DECISION_MODEL_GIT_DIRTY": dirty,
                     "SAGEMAKER_SUPPRESS_V2_WARNING": "1"},
        tags=[{"Key": k, "Value": v} for k, v in {**launcher.DEFAULT_TAGS, "purpose": "bench-queue"}.items()],
    )
    estimator.fit(wait=False)
    print(f"queue job: {estimator.latest_training_job.name} (log: CloudWatch /aws/sagemaker/TrainingJobs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--items", required=True, help='"model:GROUP+GROUP,model:GROUP", in submit order')
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--role", required=True)
    parser.add_argument("--image-uri", required=True, help="training image of the load-test jobs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-run-hours", type=int, default=4, help="per load-test job")
    parser.add_argument("--region", default="ap-northeast-2")
    parser.add_argument("--ondemand-slots", type=int, default=1, help="on-demand quota of the serving instance")
    parser.add_argument("--spot-slots", type=int, default=1, help="spot quota of the serving instance")
    parser.add_argument("--max-hours", type=float, default=10, help="give up submitting after this long")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--label", default=None, help="measurement round, e.g. r2 (bench/job/launch.py --label)")
    parser.add_argument("--remote", action="store_true", help="run the queue as a CPU SageMaker job instead")
    args = parser.parse_args()
    if ";" in args.items:
        parser.error("separate items with commas (a semicolon ends the command in the SageMaker job's shell)")
    parse_items(args.items)   # fail early on a malformed list
    if args.remote:
        submit_remote(args)
        return
    if os.environ.get("SM_TRAINING_ENV"):
        # Inside the queue job: the launcher needs the SageMaker SDK (the CPU image does not have it).
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sagemaker>=2.200,<3", "pyyaml"], check=True)
    run_queue(args)


if __name__ == "__main__":
    main()
