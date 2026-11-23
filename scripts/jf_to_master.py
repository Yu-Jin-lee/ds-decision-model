#!/usr/bin/env python3
"""Turn a Journey Finder task CSV (jf_samples/<industry>/tasks/s5.csv) into task-master rows.

    python scripts/jf_to_master.py --task s5 --csv jf_samples/beauty_cosmetics/tasks/s5.csv --country kr \
        --out jf_s5_kr.jsonl.gz
    python scripts/prepare_data.py --task s5 --master jf_s5_kr.jsonl.gz --out data_ab

The rows take the same shape as the training masters (see tests/test_prepare.py), so prepare_data.py and
configs/tasks.yaml turn them into exactly the model input, questions and label set the models were trained and
evaluated with: target / intent / site as lists, the SERP as {ai_overview, features}, brand as kind
brand / generic, id "<country>:<keyword>". Every row goes to the development partition.

The label is the existing system's answer (JF's Gemma output), not a human label: scored against it,
"accuracy" is agreement with that system.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

LANG = {"kr": "ko", "jp": "ja", "us": "en"}


def split_list(value: str) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


def serp(value: str) -> dict | None:
    """JF CSV SERP ({"ai_overview": text} or {"results": [{title, snippet}]}) in the master's shape."""
    if not value:
        return None
    raw = json.loads(value)
    out: dict = {}
    if raw.get("ai_overview"):
        out["ai_overview"] = raw["ai_overview"]
    if raw.get("results"):
        out["features"] = raw["results"]
    return out


def s5_row(row: dict[str, str]) -> dict:
    return {
        "id": f"{row['country']}:{row['keyword']}",
        "lang": LANG[row["country"]],
        "partition": "development",
        "keyword": row["keyword"],
        "kind": "brand" if row["brand"].strip().lower() == "true" else "generic",
        "stage": row["cdj_stage"],
        "stage_source": "jf_gemma",
        "industry": row["industry"],
        "category": row["category"],
        "target": split_list(row["target"]),
        "intent": split_list(row["intent"]),
        "site": split_list(row["site"]),
        "serp": serp(row["serp"]),
        "meta": {"type": None, "hard_type": None, "kind_changed": None},
        "context": {"job_id": row["job_id"], "serp_type": row.get("serp_type"), "source": "jf_samples"},
    }


ROW = {"s5": s5_row}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=sorted(ROW))
    parser.add_argument("--csv", required=True)
    parser.add_argument("--country", action="append", help="kr, jp or us (repeatable); all when left out")
    parser.add_argument("--out", required=True, help="master .jsonl or .jsonl.gz")
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    with open(args.csv, encoding="utf-8-sig", newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if not args.country or r["country"] in args.country]
    masters = [ROW[args.task](r) for r in rows]
    ids = [m["id"] for m in masters]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate ids: {len(ids) - len(set(ids))}")
    opener = gzip.open if args.out.endswith(".gz") else open
    with opener(args.out, "wt", encoding="utf-8") as handle:
        for master in masters:
            handle.write(json.dumps(master, ensure_ascii=False) + "\n")
    print(f"{len(masters)} rows -> {args.out}")


if __name__ == "__main__":
    main()
