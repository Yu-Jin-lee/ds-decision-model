"""Canonical data layout and record building.

Canonical records live at ``{data_root}/{task}/{lang}/{partition}.jsonl``,
one record per keyword::

    {"id": ..., "task": "s4", "lang": "ko", "partition": "train",
     "inputs": {...}, "label": "brand", "variant": null}

``inputs`` holds the raw input fields; the model's ``state`` is assembled from
them when the data is read, using the task's ``state`` spec (see inputs.py).
Older records with a ready-made ``state`` are used as they are.

``scripts/prepare_data.py`` writes this layout from the task masters. Every
backend then builds its own training file from canonical records, so the same
data feeds every model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

from .formats import TaskSpec, build_questions
from .inputs import assemble_state

PARTITIONS = ("train", "calibration", "development", "test")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cell_path(data_root: str | Path, task: str, lang: str, partition: str) -> Path:
    return Path(data_root) / task / lang / f"{partition}.jsonl"


def load_cells(
    data_root: str | Path,
    tasks: Iterable[str],
    langs: Iterable[str],
    partition: str,
) -> Iterator[dict[str, Any]]:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}")
    for task in tasks:
        for lang in langs:
            path = cell_path(data_root, task, lang, partition)
            if not path.exists():
                raise FileNotFoundError(f"missing data cell {path}")
            yield from read_jsonl(path)


def to_model_record(
    record: dict[str, Any],
    spec: TaskSpec,
    level: str,
    with_labels: bool,
) -> dict[str, Any]:
    """Kev/SystemOne-shaped record. Keeps id, task, lang and gold for scoring."""
    return {
        "id": record["id"],
        "task": record["task"],
        "lang": record["lang"],
        "level": level,
        "variant": record.get("variant"),
        "gold": record["label"],
        "state": record["state"] if "state" in record else assemble_state(record["inputs"], record.get("variant"), spec.raw),
        "questions": build_questions(spec, level, record, with_labels=with_labels),
    }


def build_records(
    data_root: str | Path,
    specs: dict[str, TaskSpec],
    levels: dict[str, str],
    langs: Iterable[str],
    partition: str,
    with_labels: bool,
) -> list[dict[str, Any]]:
    langs = list(langs)
    records = []
    for task, level in levels.items():
        for record in load_cells(data_root, [task], langs, partition):
            records.append(to_model_record(record, specs[task], level, with_labels))
    return records


def data_manifest(data_root: str | Path) -> dict[str, str]:
    """sha256 of every canonical file, recorded with each run."""
    root = Path(data_root)
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*.jsonl"))}
