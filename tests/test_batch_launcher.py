"""Batch job names show which runs are inside, within SageMaker's length limit."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as Run

spec = importlib.util.spec_from_file_location(
    "launch", Path(__file__).resolve().parents[1] / "scripts" / "launch_sagemaker.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)


def runs(*items):
    return [Run(group=g, seed=s) for g, s in items]


def test_batch_name_lists_groups_and_seeds():
    name = launch.batch_job_name("kev-4b", runs(
        ("B0-s5-score", 0), ("S-s5-choice", 0), ("S-s5-score", 0), ("S-s5-both", 0)))
    assert name == "kev-4b-batch-b0s5sc-s5chscbo-s0"
    name = launch.batch_job_name("laya-multilingual", runs(("S-s5-choice", 1), ("S-s5-both", 2)))
    assert name == "laya-multilingual-batch-s5chbo-s12"


def test_batch_name_falls_back_when_too_long():
    many = runs(*[(g, s) for g in ("S-s2-choice", "S-s2-split", "S-s4-noul", "S-s5-choice", "J+R-ko", "J+R-ja")
                  for s in (0, 1, 2)])
    name = launch.batch_job_name("laya-multilingual", many)
    assert name.startswith("laya-multilingual-batch-18runs-") and len(name) <= launch.MAX_BASE_NAME
