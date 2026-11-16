#!/usr/bin/env python3
"""Write canonical data cells from a task master.

    python scripts/prepare_data.py --task s4 --master s4_master.jsonl --out data

The master has one row per keyword with ``id``, ``lang``, ``partition`` and the
task's state fields and label field (see configs/tasks.yaml). Partitions are
taken from the master as fixed by the data-building documents (industry split);
this script does not re-split. Output: ``{out}/{task}/{lang}/{partition}.jsonl``
plus ``{out}/manifest.json`` with the sha256 of every file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_model.config import CONFIG_DIR, load_yaml  # noqa: E402
from decision_model.data import PARTITIONS, data_manifest, read_jsonl, write_jsonl  # noqa: E402

LANG_ALIASES = {"kr": "ko", "jp": "ja", "us": "en"}


def canonical(row: dict, task: str, spec: dict) -> dict:
    raw_label = str(row[spec.get("label_field", "label")])
    label = spec.get("label_map", {}).get(raw_label, raw_label)
    if label not in spec["final_labels"]:
        raise ValueError(f"{row.get('id')}: label {label!r} not in {spec['final_labels']}")
    variant = None
    if spec.get("variant_field"):
        variant = "brand" if row[spec["variant_field"]] in (True, "brand", "브랜드", 1, "1", "true") else "nonbrand"
    state = {field: row.get(field) for field in spec["state_fields"] if field != spec.get("variant_field")}
    if variant is not None:
        state["brand"] = variant == "brand"
    lang = LANG_ALIASES.get(row["lang"], row["lang"])
    return {
        "id": row["id"],
        "task": task,
        "lang": lang,
        "partition": row["partition"],
        "state": state,
        "label": label,
        "variant": variant,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--master", required=True, help="master JSONL for the task")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    spec = load_yaml(CONFIG_DIR / "tasks.yaml")["tasks"][args.task]
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in read_jsonl(args.master):
        record = canonical(row, args.task, spec)
        if record["partition"] not in PARTITIONS:
            raise ValueError(f"{record['id']}: unknown partition {record['partition']!r}")
        cells[(record["lang"], record["partition"])].append(record)

    out = Path(args.out)
    for (lang, partition), records in sorted(cells.items()):
        count = write_jsonl(out / args.task / lang / f"{partition}.jsonl", records)
        shares = Counter(r["label"] for r in records)
        low = [label for label in spec["final_labels"] if shares[label] / count < 0.05]
        warning = f"  labels under 5%: {low}" if low else ""
        print(f"{args.task}/{lang}/{partition}: {count}{warning}")
    (out / "manifest.json").write_text(json.dumps(data_manifest(out), indent=2))


if __name__ == "__main__":
    main()
