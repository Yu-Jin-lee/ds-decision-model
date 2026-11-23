#!/usr/bin/env python3
"""Submit scripts/ab_job.py as a SageMaker training job (one finished run, one outside data set).

    python scripts/ab_launch.py --model chon-1.0-e4b --group J-best --seed 0 --task s5 --lang ko \
        --name jf-beauty-kr-s5 --data s3://.../data-ab/jf-beauty-kr-s5/ --output s3://.../runs/ \
        --role ... --image-uri ... --spot

Same image and instance type as the runs (ml.g6e.xlarge by default). --spot uses the separate spot quota, so it
can run next to an on-demand job. Results: {output}ab/{name}/{run_id}/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from launch_sagemaker import DEFAULT_TAGS, git_state  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("--model", "--group", "--task", "--lang", "--name", "--data", "--output", "--role", "--image-uri"):
        parser.add_argument(name, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--instance-type", default="ml.g6e.xlarge")
    parser.add_argument("--volume-gb", type=int, default=100)
    parser.add_argument("--spot", action="store_true")
    parser.add_argument("--max-run-hours", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    commit, dirty = git_state()
    output = args.output if args.output.endswith("/") else args.output + "/"
    base = f"{args.model}-ab-{args.name}"[:38].rstrip("-")
    hyper = {"model": args.model, "group": args.group, "seed": str(args.seed), "task": args.task,
             "lang": args.lang, "name": args.name}
    print(f"submit {args.model}/{args.group}/s{args.seed} {args.task}/{args.lang} on {args.name} -> {base} "
          f"({args.instance_type}{' spot' if args.spot else ''}, commit {commit[:7]}{' dirty' if dirty == '1' else ''})")
    if args.dry_run:
        return
    from sagemaker.estimator import Estimator

    max_run = args.max_run_hours * 3600
    estimator = Estimator(
        image_uri=args.image_uri, role=args.role, instance_type=args.instance_type, instance_count=1,
        volume_size=args.volume_gb, entry_point="scripts/ab_job.py", source_dir=str(REPO), hyperparameters=hyper,
        output_path=f"{output}_jobs/", code_location=f"{output}_code", base_job_name=base,
        use_spot_instances=args.spot, max_run=max_run, max_wait=max_run * 2 if args.spot else None,
        environment={"SM_OUTPUT_PREFIX": output, "DECISION_MODEL_GIT_COMMIT": commit,
                     "DECISION_MODEL_GIT_DIRTY": dirty},
        tags=[{"Key": k, "Value": v} for k, v in {**DEFAULT_TAGS, "kind": "ab"}.items()],
    )
    estimator.fit({"data": args.data}, wait=False)
    print(f"submitted {estimator.latest_training_job.name}")


if __name__ == "__main__":
    main()
