#!/usr/bin/env python3
"""Method ``endpoint``: deploy one run as a SageMaker real-time endpoint, load-test it with Locust through
InvokeEndpoint, then delete the endpoint (also when something fails).

Run it from a machine that is not the endpoint: an EC2 instance in the same region (ap-northeast-2) keeps the
network out of the numbers as much as a real client would; a laptop adds the internet round trip. Needs AWS
credentials that may create SageMaker models/endpoints and pass the role, and the data locally (records).

    python bench/endpoint/run.py --model kev-4b --group B0-s5-choice --seed 0 \
        --image-uri <ECR image from build_image.sh> --role <SageMaker role ARN> \
        --output s3://BUCKET/ds-jf-decision-model/runs/ --upload
    python bench/endpoint/run.py --model kev-4b --group B0-s5-choice --seed 0 --name <endpoint already up>

Results: {out-root}/bench/endpoint/{model}/{group}/s{seed}/ (same files as method job); --upload also copies
them to {output}/bench/endpoint/... Compare with bench/compare.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

# Locust patches the standard library for gevent; import it before anything opens sockets.
from decision_model.load_test import SageMakerEndpointUser, run_sweep  # noqa: E402,I001

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from typing import Any  # noqa: E402

import deploy  # noqa: E402
from decision_model.bench import (  # noqa: E402
    build_plan, describe_plan, load_bench_config, make_summary, result_dir, write_results,
)
from decision_model.pipeline import find_run  # noqa: E402


def endpoint_info(runtime: Any, name: str) -> dict[str, Any]:
    """What GET /info answers, asked through InvokeEndpoint (decision_model.endpoint)."""
    response = runtime.invoke_endpoint(EndpointName=name, Body=b'{"__info__": true}', ContentType="application/json")
    return json.loads(response["Body"].read())


def upload(out_dir: Path, out_root: Path, output: str, region: str) -> None:
    """Copy the result folder to {output}/bench/endpoint/..., summary.json last."""
    import boto3

    bucket, _, prefix = output.removeprefix("s3://").partition("/")
    client = boto3.client("s3", region_name=region)
    files = sorted((p for p in out_dir.rglob("*") if p.is_file()), key=lambda p: p.name == "summary.json")
    for path in files:
        key = f"{prefix.rstrip('/')}/{path.relative_to(out_root).as_posix()}".lstrip("/")
        client.upload_file(str(path), bucket, key)
    print(f"uploaded {len(files)} files to s3://{bucket}/{prefix.rstrip('/')}/{out_dir.relative_to(out_root).as_posix()}")


def main() -> None:
    config = load_bench_config()
    settings = config["endpoint"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--group", required=True, help="expanded group name; picks the task(s), level and languages")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-uri", default=None, help="endpoint image (bench/endpoint/build_image.sh)")
    parser.add_argument("--role", default=None, help="SageMaker execution role ARN for the endpoint")
    parser.add_argument("--output", default=None, help="S3 runs prefix: trained runs' checkpoints, and --upload")
    parser.add_argument("--name", default=None, help="measure an endpoint that is already up (not deployed or deleted)")
    parser.add_argument("--keep", action="store_true", help="do not delete the endpoint afterwards (it keeps costing)")
    parser.add_argument("--upload", action="store_true", help="also copy the results to --output")
    parser.add_argument("--region", default=settings["region"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--out-root", default="runs")
    parser.add_argument("--label", default=None)
    parser.add_argument("--users", default=None, help="comma-separated, e.g. 1,4,16 (default: configs/bench.yaml)")
    parser.add_argument("--requests-per-step", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    args = parser.parse_args()
    if not args.name and not (args.image_uri and args.role):
        parser.error("give --image-uri and --role to deploy, or --name of an endpoint that is up")
    if args.upload and not args.output:
        parser.error("--upload needs --output")

    run = find_run(args.model, args.group, args.seed)
    if run.train and not args.name and not args.output:
        parser.error(f"{run.run_id} is a trained run: --output (runs prefix) is needed to find its checkpoint")
    plan = build_plan(run, args.data_root, config, [int(u) for u in args.users.split(",")] if args.users else None,
                      args.requests_per_step, args.warmup)
    print(describe_plan(run.run_id, plan), flush=True)

    import boto3

    sagemaker = boto3.client("sagemaker", region_name=args.region)
    runtime = boto3.client("sagemaker-runtime", region_name=args.region)
    name = args.name or deploy.endpoint_name(run.model, run.group, run.seed)
    instance = deploy.instance_type()
    started = time.time()
    deployed_seconds = None
    try:
        if not args.name:
            deploy.deploy(sagemaker, name, args.image_uri, args.role,
                          deploy.container_environment(run.model, run.group, run.seed, args.output), instance,
                          settings["startup_timeout_s"], {**deploy.TAGS, "run": run.run_id},
                          log=lambda line: print(line, flush=True))
            deployed_seconds = round(time.time() - started, 1)
        described = sagemaker.describe_endpoint(EndpointName=name)
        info = endpoint_info(runtime, name)
        print(f"endpoint {name}: {json.dumps({k: v for k, v in info.items() if k != 'environment'}, ensure_ascii=False)}",
              flush=True)
        steps, rows = run_sweep(plan, config["max_step_seconds"], SageMakerEndpointUser,
                                log=lambda line: print(line, flush=True), endpoint_name=name, region=args.region)
    finally:
        if not args.name and not args.keep:
            deploy.teardown(sagemaker, name, log=lambda line: print(line, flush=True))
        elif not args.name:
            print(f"kept {name}; delete it when done: python bench/endpoint/deploy.py down --name {name}")

    deployment = {
        "endpoint_name": name,
        "instance_type": instance,
        "region": args.region,
        "image_uri": args.image_uri or described["ProductionVariants"][0].get("DeployedImages", [{}])[0].get("SpecifiedImage"),
        "deployed_here": not args.name,
        "deploy_seconds": deployed_seconds,
        "deleted": not args.name and not args.keep,
    }
    summary = make_summary("endpoint", run, plan, steps, info, config, args.label, None,
                           deployment=deployment, client_on_endpoint_host=False,
                           seconds=round(time.time() - started, 1))
    out_root = Path(args.out_root)
    out_dir = result_dir(out_root, "endpoint", run.run_id, args.label)
    write_results(out_dir, summary, rows)
    print(f"wrote {out_dir}", flush=True)
    if args.upload:
        upload(out_dir, out_root, args.output, args.region)
    print(json.dumps(summary["best"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
