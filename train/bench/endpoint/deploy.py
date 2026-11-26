#!/usr/bin/env python3
"""Create and delete the SageMaker real-time endpoint of one run (method ``endpoint``).

bench/endpoint/run.py calls these itself (deploy, measure, delete). Use this script directly to keep an
endpoint up for a while or to clean up:

    python bench/endpoint/deploy.py up --model kev-4b --group B0-s5-choice --seed 0 --image-uri <ECR image> \
        --role <SageMaker role ARN> --output s3://BUCKET/ds-jf-decision-model/runs/
    python bench/endpoint/deploy.py down --name <endpoint name>
    python bench/endpoint/deploy.py list          # load-test endpoints still up (they cost money while up)

One endpoint = SageMaker model + endpoint config + endpoint with the same name, on the serving instance
(configs/serving.yaml), one instance, tagged like the training jobs plus purpose=bench.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from decision_model.config import CONFIG_DIR, load_yaml  # noqa: E402

TAGS = {"project": "ds-jf-decision-model", "team": "ds", "purpose": "bench"}
VARIANT = "AllTraffic"


def endpoint_name(model: str, group: str, seed: int, now: float | None = None) -> str:
    """``dm-bench-kev-4b-b0-s5-choice-s0-1126153012``: at most 63 characters, letters, digits and hyphens."""
    stamp = time.strftime("%m%d%H%M%S", time.gmtime(now))
    base = re.sub(r"[^a-zA-Z0-9-]+", "-", f"dm-bench-{model}-{group}-s{seed}").strip("-").lower()
    return f"{base[:63 - len(stamp) - 1].rstrip('-')}-{stamp}"


def instance_type() -> str:
    return load_yaml(CONFIG_DIR / "serving.yaml")["instance"]


def container_environment(model: str, group: str, seed: int, runs_prefix: str | None) -> dict[str, str]:
    """What bench/endpoint/serve.py reads (the model name is already in the image)."""
    env = {"DECISION_MODEL_NAME": model, "BENCH_GROUP": group, "BENCH_SEED": str(seed)}
    if runs_prefix:
        env["SM_OUTPUT_PREFIX"] = runs_prefix
    return env


def deploy(sagemaker: Any, name: str, image_uri: str, role: str, environment: dict[str, str], instance: str,
           startup_timeout_s: int, tags: dict[str, str], log=print, wait: bool = True) -> None:
    """Create model, endpoint config and endpoint ``name``; wait until it is InService. If anything fails,
    whatever was created is deleted before the error is raised."""
    tag_list = [{"Key": k, "Value": v} for k, v in tags.items()]
    try:
        sagemaker.create_model(ModelName=name, ExecutionRoleArn=role, Tags=tag_list,
                               PrimaryContainer={"Image": image_uri, "Environment": environment, "Mode": "SingleModel"})
        sagemaker.create_endpoint_config(EndpointConfigName=name, Tags=tag_list, ProductionVariants=[{
            "VariantName": VARIANT, "ModelName": name, "InstanceType": instance, "InitialInstanceCount": 1,
            "ContainerStartupHealthCheckTimeoutInSeconds": int(startup_timeout_s),
        }])
        sagemaker.create_endpoint(EndpointName=name, EndpointConfigName=name, Tags=tag_list)
        log(f"creating endpoint {name} on {instance} ({image_uri})")
        if wait:
            wait_in_service(sagemaker, name, startup_timeout_s + 1800, log)
    except BaseException:
        log(f"deploying {name} failed; deleting what was created")
        teardown(sagemaker, name, log)
        raise


def wait_in_service(sagemaker: Any, name: str, timeout_s: float, log=print, poll_s: float = 30) -> None:
    deadline = time.time() + timeout_s
    last = None
    while time.time() < deadline:
        described = sagemaker.describe_endpoint(EndpointName=name)
        status = described["EndpointStatus"]
        if status != last:
            log(f"endpoint {name}: {status}")
            last = status
        if status == "InService":
            return
        if status in ("Failed", "OutOfService", "Deleting"):
            raise RuntimeError(f"endpoint {name} is {status}: {described.get('FailureReason', '')}")
        time.sleep(poll_s)
    raise TimeoutError(f"endpoint {name} was not InService after {timeout_s:.0f} s")


def _not_found(error: Exception) -> bool:
    code = getattr(error, "response", {}).get("Error", {}).get("Code", "")
    message = str(error)
    return code in ("ValidationException", "ResourceNotFound") and ("Could not find" in message or "does not exist" in message)


def _settle(sagemaker: Any, name: str, log=print, timeout_s: float = 5400, poll_s: float = 30) -> None:
    """An endpoint that is still Creating or Updating cannot be deleted yet: wait until it is not."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = sagemaker.describe_endpoint(EndpointName=name)["EndpointStatus"]
        except Exception:  # noqa: BLE001 - gone, or not created
            return
        if status not in ("Creating", "Updating", "SystemUpdating"):
            return
        log(f"endpoint {name} is {status}; waiting to delete it")
        time.sleep(poll_s)


def teardown(sagemaker: Any, name: str, log=print) -> list[str]:
    """Delete endpoint, endpoint config and model ``name`` (each one that exists). Returns what failed."""
    _settle(sagemaker, name, log)
    failed = []
    for kind, call in (("endpoint", lambda: sagemaker.delete_endpoint(EndpointName=name)),
                       ("endpoint config", lambda: sagemaker.delete_endpoint_config(EndpointConfigName=name)),
                       ("model", lambda: sagemaker.delete_model(ModelName=name))):
        try:
            call()
            log(f"deleted {kind} {name}")
        except Exception as error:  # noqa: BLE001 - try the others too
            if not _not_found(error):
                log(f"could not delete {kind} {name}: {error}")
                failed.append(kind)
    return failed


def list_bench_endpoints(sagemaker: Any) -> list[dict[str, Any]]:
    endpoints = []
    for page in sagemaker.get_paginator("list_endpoints").paginate(NameContains="dm-bench-"):
        endpoints += page["Endpoints"]
    return endpoints


def _client(region: str) -> Any:
    import boto3

    return boto3.client("sagemaker", region_name=region)


def main() -> None:
    config = load_yaml(CONFIG_DIR / "bench.yaml")["endpoint"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", default=config["region"])
    sub = parser.add_subparsers(dest="command", required=True)
    up = sub.add_parser("up", help="create an endpoint and wait until it is InService")
    up.add_argument("--model", required=True)
    up.add_argument("--group", required=True)
    up.add_argument("--seed", type=int, default=0)
    up.add_argument("--image-uri", required=True, help="from bench/endpoint/build_image.sh")
    up.add_argument("--role", required=True)
    up.add_argument("--output", default=None, help="S3 runs prefix (needed for a trained run's checkpoint)")
    down = sub.add_parser("down", help="delete an endpoint, its config and its model")
    down.add_argument("--name", required=True)
    sub.add_parser("list", help="load-test endpoints that exist")
    args = parser.parse_args()

    sagemaker = _client(args.region)
    if args.command == "up":
        name = endpoint_name(args.model, args.group, args.seed)
        deploy(sagemaker, name, args.image_uri, args.role,
               container_environment(args.model, args.group, args.seed, args.output), instance_type(),
               config["startup_timeout_s"], {**TAGS, "run": f"{args.model}/{args.group}/s{args.seed}"})
        print(f"{name} is InService; delete it when done: python bench/endpoint/deploy.py down --name {name}")
    elif args.command == "down":
        sys.exit(1 if teardown(sagemaker, args.name) else 0)
    else:
        for endpoint in list_bench_endpoints(sagemaker):
            print(f"{endpoint['EndpointName']}\t{endpoint['EndpointStatus']}\t{endpoint['CreationTime']}")


if __name__ == "__main__":
    main()
