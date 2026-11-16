#!/usr/bin/env python3
"""Run one experiment cell (model x group x seed), or list the run matrix.

Locally or inside a SageMaker training job:

    python scripts/run.py --model clef-flash --group S-s4-noul --seed 0
    python scripts/run.py --model kev-9b --list --stage 1

Inside SageMaker the data channel and output directory come from the
environment (SM_CHANNEL_DATA, SM_MODEL_DIR) unless given explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_model.config import CONFIG_DIR, expand_runs, load_yaml  # noqa: E402
from decision_model.pipeline import execute, find_run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="config name under configs/models")
    parser.add_argument("--group", help="expanded group name, e.g. S-s4-noul or J+R-all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-root", default=os.environ.get("SM_CHANNEL_DATA", "data"))
    parser.add_argument("--out-root", default=os.environ.get("SM_MODEL_DIR", "runs"))
    parser.add_argument("--replay-file", default=None, help="Kev-format JSONL for replay (non-Kev backends)")
    parser.add_argument("--final", action="store_true", help="also score the locked test partition (once)")
    parser.add_argument("--list", action="store_true", help="print the run matrix and exit")
    parser.add_argument("--stage", type=int, action="append", help="with --list: only these stages")
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    if args.list:
        runs = expand_runs(
            args.model,
            load_yaml(CONFIG_DIR / "groups.yaml"),
            load_yaml(CONFIG_DIR / "tasks.yaml"),
            stages=set(args.stage) if args.stage else None,
            include_optional=args.include_optional,
        )
        for run in runs:
            print(json.dumps(run.to_dict(), ensure_ascii=False))
        print(f"# {len(runs)} runs ({sum(r.train for r in runs)} with training)", file=sys.stderr)
        return

    if not args.group:
        parser.error("--group is required unless --list")
    run = find_run(args.model, args.group, args.seed)
    replay_file = Path(args.replay_file) if args.replay_file else None
    if replay_file is None and (Path(args.data_root) / "replay.jsonl").exists():
        replay_file = Path(args.data_root) / "replay.jsonl"
    summary = execute(run, Path(args.data_root), Path(args.out_root), final=args.final, replay_file=replay_file)
    print(json.dumps({k: summary[k] for k in ("temperature", "metrics_development")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
