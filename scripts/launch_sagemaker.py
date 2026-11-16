#!/usr/bin/env python3
"""Submit SageMaker training jobs for the run matrix.

    python scripts/launch_sagemaker.py --model clef-flash --stage 1 \
        --data s3://BUCKET/decision-model/data/ --output s3://BUCKET/decision-model/runs/ \
        --role arn:aws:iam::ACCOUNT:role/SageMakerExecutionRole --image-uri <ECR image>

Each job runs one (group, seed). Runs whose summary.json already exists under
--output are skipped, so the command can be re-run after a failure.
Use --dry-run first to see the jobs it would submit.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_model.config import CONFIG_DIR, expand_runs, load_model_config, load_yaml  # noqa: E402
from decision_model.pipeline import _base_group  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# Cost-allocation tags put on every training job. The S3 bucket carries the same
# tags, so Cost Explorer can show this project's spend once they are activated.
DEFAULT_TAGS = {"project": "ds-jf-decision-model", "team": "ds"}


def job_name(model: str, group: str, seed: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9-]", "-", f"{model}-{group}-s{seed}").strip("-")
    return slug[:50]


def git_state() -> tuple[str, str]:
    """Commit and dirty flag of the local checkout, recorded in each run's summary."""
    import subprocess

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown", "1"
    return commit, "1" if dirty else "0"


def already_done(s3_output: str, run_id: str) -> bool:
    import boto3

    bucket, _, prefix = s3_output.removeprefix("s3://").partition("/")
    key = f"{prefix.rstrip('/')}/{run_id}/summary.json".lstrip("/")
    response = boto3.client("s3").list_objects_v2(Bucket=bucket, Prefix=key, MaxKeys=1)
    return response.get("KeyCount", 0) > 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--stage", type=int, action="append")
    parser.add_argument(
        "--group", action="append",
        help="config group (e.g. S-s5) or one expanded group (e.g. S-s5-choice); repeatable",
    )
    parser.add_argument("--seed", type=int, action="append", help="only these seeds (e.g. --seed 0 for a smoke run)")
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--data", required=True, help="S3 prefix of the canonical data")
    parser.add_argument("--output", required=True, help="S3 prefix for run outputs")
    parser.add_argument("--role", required=True)
    parser.add_argument("--image-uri", required=True, help="training image (one image for every run)")
    parser.add_argument("--instance-type", default=None, help="defaults to the model config's instance")
    parser.add_argument("--spot", action="store_true", help="managed spot training")
    parser.add_argument("--max-run-hours", type=int, default=12)
    parser.add_argument("--tag", action="append", default=[], help="extra/override tag key=value")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    model_config = load_model_config(args.model)
    groups_config = load_yaml(CONFIG_DIR / "groups.yaml")
    base_groups = {_base_group(g, groups_config) for g in args.group} if args.group else None
    runs = expand_runs(
        args.model,
        groups_config,
        load_yaml(CONFIG_DIR / "tasks.yaml"),
        stages=set(args.stage) if args.stage else None,
        groups=base_groups,
        include_optional=args.include_optional,
    )
    if args.group:
        wanted = set(args.group)
        runs = [run for run in runs if run.group in wanted or _base_group(run.group, groups_config) in wanted]
    if args.seed:
        runs = [run for run in runs if run.seed in set(args.seed)]
    if not runs:
        sys.exit("no runs match the given --stage/--group/--seed")
    instance_type = args.instance_type or model_config["instance"]
    tags = dict(DEFAULT_TAGS)
    for item in args.tag:
        key, sep, value = item.partition("=")
        if not sep or not key:
            sys.exit(f"--tag must be key=value, got {item!r}")
        tags[key] = value
    tag_list = [{"Key": k, "Value": v} for k, v in tags.items()]
    commit, dirty = git_state()
    if dirty == "1":
        print(f"warning: uncommitted changes; runs will record commit {commit[:7]} with git_dirty=true")
    output = args.output.rstrip("/")

    for run in runs:
        if not args.dry_run and already_done(args.output, run.run_id):
            print(f"skip   {run.run_id} (summary.json exists)")
            continue
        name = job_name(run.model, run.group, run.seed)
        if args.dry_run:
            print(f"submit {run.run_id} -> {name} on {instance_type} commit={commit[:7]} tags={tags}")
            continue
        from sagemaker.estimator import Estimator

        max_run = args.max_run_hours * 3600
        estimator = Estimator(
            image_uri=args.image_uri,
            role=args.role,
            instance_type=instance_type,
            instance_count=1,
            entry_point="sagemaker/entry.py",
            source_dir=str(REPO),
            hyperparameters={"model": run.model, "group": run.group, "seed": run.seed},
            output_path=f"{output}/_jobs/",
            code_location=f"{output}/_code",
            checkpoint_s3_uri=None,
            base_job_name=name,
            use_spot_instances=args.spot,
            max_run=max_run,
            max_wait=max_run * 2 if args.spot else None,
            environment={
                "SM_OUTPUT_PREFIX": args.output,
                "DECISION_MODEL_GIT_COMMIT": commit,
                "DECISION_MODEL_GIT_DIRTY": dirty,
            },
            tags=tag_list,
        )
        estimator.fit({"data": args.data}, wait=False)
        print(f"submit {run.run_id} -> {estimator.latest_training_job.name}")


if __name__ == "__main__":
    main()
