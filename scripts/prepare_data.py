#!/usr/bin/env python3
"""Write canonical data cells from a task master.

    python scripts/prepare_data.py --task s5 --master s5_master.jsonl.gz --out data

The master has one row per keyword with ``id``, ``lang`` and ``partition``
(``part`` also works); which fields become the model input, the label and the
brand / non-brand variant is set per task in configs/tasks.yaml. Partitions are
taken from the master as fixed by the data-building documents (industry split);
this script does not re-split. It stops without writing anything if any row
cannot be converted.

Output: ``{out}/{task}/{lang}/{partition}.jsonl`` and ``{out}/manifest.json``
with the sha256 of every file.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from decision_model.config import CONFIG_DIR, load_yaml  # noqa: E402
from decision_model.data import PARTITIONS, data_manifest, write_jsonl  # noqa: E402
from decision_model.prepare import convert, read_master  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True)
    parser.add_argument("--master", required=True, help="master JSONL (.jsonl or .jsonl.gz)")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    spec = load_yaml(CONFIG_DIR / "tasks.yaml")["tasks"][args.task]
    records, errors = convert(read_master(args.master), args.task, spec)
    bad_partition = [r["id"] for r in records if r["partition"] not in PARTITIONS]
    errors += [f"{i}: unknown partition" for i in bad_partition]
    if errors:
        print(f"{len(errors)} rows could not be converted; nothing written:", file=sys.stderr)
        for message in errors[:30]:
            print("  " + message, file=sys.stderr)
        sys.exit(1)

    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        cells[(record["lang"], record["partition"])].append(record)

    out = Path(args.out)
    for (lang, partition), cell in sorted(cells.items()):
        count = write_jsonl(out / args.task / lang / f"{partition}.jsonl", cell)
        shares = Counter(r["label"] for r in cell)
        low = [f"{label} {shares[label] / count:.1%}" for label in spec["final_labels"] if shares[label] / count < 0.05]
        warning = f"  labels under 5%: {', '.join(low)}" if low else ""
        print(f"{args.task}/{lang}/{partition}: {count}{warning}")
    (out / "manifest.json").write_text(json.dumps(data_manifest(out), indent=2))
    print(f"wrote {len(records)} records, manifest at {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
