#!/usr/bin/env python3
"""Method ``job``: submit SageMaker training jobs that load-test finished runs (configs/bench.yaml).

Each job takes the serving instance (configs/serving.yaml, one for every model), installs the model's serving
engine and Locust, serves the model on localhost (bench/serve_endpoint.py), runs bench/job/run.py against it,
and uploads bench/job/{run_id}/ to --output. Trained runs read their checkpoint from --output; B0 and P0 runs
serve the released model. The instance is released when the job ends.

    python bench/job/launch.py --model gemma-4-26b-a4b --group P0-s5 --seed 0 \
        --data s3://BUCKET/ds-jf-decision-model/data/ --output s3://BUCKET/ds-jf-decision-model/runs/ \
        --role <SageMaker role ARN> --image-uri <training image> --dry-run
    python bench/job/launch.py --model kev-4b --group B0-s5-choice --group S-s5-choice --seed 0 --batch ...

Options are the same as scripts/launch_sagemaker.py (without --serving). Results already in --output are skipped.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _launcher():
    spec = importlib.util.spec_from_file_location("launch_sagemaker", REPO / "scripts" / "launch_sagemaker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    _launcher().main(sys.argv[1:], bench=True, description=__doc__)
