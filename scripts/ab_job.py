#!/usr/bin/env python3
"""A/B check: score one finished run's model on outside data, set up exactly as the run was evaluated.

    python scripts/ab_job.py --model chon-1.0-e4b --group J-best --seed 0 --task s5 --lang ko \
        --name jf-beauty-kr-s5 --data-root data_ab --runs-prefix s3://BUCKET/ds-jf-decision-model/runs/

Held equal to the run's own evaluation: the instructions, option text and label set of the run's level for
the task (configs/tasks.yaml), the model input format (scripts/jf_to_master.py + prepare_data.py), the run's
checkpoint and calibration temperature (read from its summary.json), and the evaluation reader (not the
serving profile). The data's labels are the existing system's answers, so "accuracy" is agreement with it.

On SageMaker (scripts/ab_launch.py) the data comes from the `data` channel and results go to
{runs-prefix}ab/{name}/{run_id}/; nothing under the run itself is touched.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "sagemaker"))


def install(config: dict) -> None:
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    subprocess.run([*pip, "-r", str(REPO / "requirements.txt")], check=True)
    if config["backend"] == "chon":
        subprocess.run([*pip, f"chon[serve] @ git+{config['chon_repo']}@{config['chon_ref']}"], check=True)
    elif config["backend"] == "laya":
        subprocess.run([*pip, "-r", str(REPO / "requirements-laya.txt")], check=True)
    else:
        raise SystemExit(f"backend {config['backend']} is not wired into ab_job.py (chon, laya)")


def write_keyword_csv(path: Path, records: list[dict], rows: list[dict]) -> None:
    """One line per keyword and question: the existing label, the model's answer and its probability."""
    import csv

    keywords = {r["id"]: r["state"].get("keyword") for r in records}
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "keyword", "variant", "view", "existing_label", "model_label", "model_prob",
                         "match", "existing_label_prob"])
        for row in rows:
            dist = row["distribution"]
            writer.writerow([row["id"], keywords.get(row["id"]), row.get("variant"), row["view"], row["gold"],
                             row["prediction"], round(dist[row["prediction"]], 4), int(row["gold"] == row["prediction"]),
                             round(dist.get(row["gold"], 0.0), 4)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", required=True)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--name", required=True, help="name of the outside data set, e.g. jf-beauty-kr-s5")
    parser.add_argument("--data-root", default=os.environ.get("SM_CHANNEL_DATA", "data_ab"))
    parser.add_argument("--runs-prefix", default=os.environ.get("SM_OUTPUT_PREFIX"))
    parser.add_argument("--out", default=os.environ.get("SM_MODEL_DIR", "runs_ab"))
    parser.add_argument("--install", type=int, default=1 if os.environ.get("SM_MODEL_DIR") else 0)
    args = parser.parse_args()
    if not args.runs_prefix:
        raise SystemExit("--runs-prefix (or SM_OUTPUT_PREFIX) is needed to read the run's summary and checkpoint")

    from decision_model.config import load_model_config

    config = load_model_config(args.model)
    if args.install:
        install(config)

    import boto3

    from decision_model.backends import get_backend
    from decision_model.config import CONFIG_DIR, load_yaml
    from decision_model.data import build_records, data_manifest, write_jsonl
    from decision_model.evaluation import score
    from decision_model.formats import load_task_specs
    from decision_model.pipeline import _git_commit, _git_dirty, find_run, save_model_inputs
    from entry import checkpoint_relative_path

    started = time.time()
    run = find_run(args.model, args.group, args.seed)
    if args.task not in run.levels:
        raise SystemExit(f"{run.run_id} was not trained on {args.task} (levels {run.levels})")
    level = run.levels[args.task]

    s3 = boto3.client("s3")
    bucket, _, key_prefix = args.runs_prefix.removeprefix("s3://").partition("/")
    run_key = f"{key_prefix.rstrip('/')}/{run.run_id}".lstrip("/")
    source = json.loads(s3.get_object(Bucket=bucket, Key=f"{run_key}/summary.json")["Body"].read())

    checkpoint = None
    if run.train:
        relative = checkpoint_relative_path(source, run.run_id)
        checkpoint = Path(os.environ.get("AB_CHECKPOINT_DIR", "/opt/ml/ab-checkpoint")) / run.run_id / relative
        count = 0
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{run_key}/{relative}/"):
            for item in page.get("Contents", []):
                target = checkpoint / item["Key"][len(f"{run_key}/{relative}/"):]
                target.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, item["Key"], str(target))
                count += 1
        if not count:
            raise SystemExit(f"no checkpoint files under s3://{bucket}/{run_key}/{relative}/")
        print(f"{run.run_id}: {count} checkpoint files -> {checkpoint}", flush=True)

    specs = load_task_specs(load_yaml(CONFIG_DIR / "tasks.yaml"))
    records = build_records(args.data_root, specs, {args.task: level}, [args.lang], "development",
                            with_labels=True, for_prediction=True)
    out = Path(args.out) / "ab" / args.name / run.run_id
    out.mkdir(parents=True, exist_ok=True)
    inputs_sha = save_model_inputs(out, "development", records)

    backend = get_backend(config)   # evaluation reader, as the run was evaluated
    predictions = backend.predict(checkpoint, records)
    rows, metrics = score(specs, records, predictions, source["temperature"])
    write_jsonl(out / "predictions_development.jsonl", predictions)
    write_jsonl(out / "scored_development.jsonl", rows)
    write_keyword_csv(out / "keywords.csv", records, rows)
    summary = {
        "name": args.name,
        "run": run.to_dict(),
        "task": args.task,
        "lang": args.lang,
        "level": level,
        "records": len(records),
        "labels": "existing system's answers: accuracy is agreement with that system",
        "source_run": {
            "summary": f"s3://{bucket}/{run_key}/summary.json",
            "git_commit": source.get("git_commit"),
            "checkpoint": source.get("checkpoint"),
            "temperature": source["temperature"],
            "metrics_development": {k: v for k, v in source.get("metrics_development", {}).items()
                                    if k.startswith(f"{args.task}/{args.lang}/")},
        },
        "inference_precision": backend.inference_precision(),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "data_manifest": data_manifest(args.data_root),
        "model_inputs_sha256": {"development": inputs_sha},
        "metrics_development": metrics,
        "seconds": round(time.time() - started, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in metrics.items()}, ensure_ascii=False)[:2000], flush=True)

    if os.environ.get("SM_MODEL_DIR"):
        dest = f"{key_prefix.rstrip('/')}/ab/{args.name}/{run.run_id}".lstrip("/")
        for path in sorted(out.rglob("*")):
            if path.is_file():
                s3.upload_file(str(path), bucket, f"{dest}/{path.relative_to(out)}")
        print(f"uploaded -> s3://{bucket}/{dest}/", flush=True)
        if checkpoint is not None:
            shutil.rmtree(checkpoint, ignore_errors=True)


if __name__ == "__main__":
    main()
