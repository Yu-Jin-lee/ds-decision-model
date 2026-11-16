"""Turn a task master's rows into canonical records (see data.py).

How a row maps to a record is configured per task in configs/tasks.yaml:

* ``state``: model input, as ``{name: path}`` (dotted paths into the row) or
  ``{name: {from: path, format: serp_compact}}``. ``state_fields`` (a list of
  top-level fields) is the short form.
* ``label_field`` and optional ``label_map``: the final label.
* ``variant_field``, ``variant_map``, ``variant_in_state``: brand / non-brand.
* ``variant_labels``: labels each variant may take.
* ``meta_fields``: kept for analysis, never shown to the model.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Iterator

LANG_ALIASES = {"kr": "ko", "jp": "ja", "us": "en"}


def read_master(path: str | Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def get_path(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def serp_compact(serp: Any) -> Any:
    """AI overview text and the top results' title and snippet; drops ids and domains."""
    if not isinstance(serp, dict):
        return serp
    out: dict[str, Any] = {}
    overview = serp.get("ai_overview")
    if isinstance(overview, dict):
        overview = overview.get("text")
    if overview:
        out["ai_overview"] = overview
    results = [
        {k: f[k] for k in ("title", "snippet") if f.get(k)}
        for f in serp.get("features") or []
        if isinstance(f, dict)
    ]
    results = [r for r in results if r]
    if results:
        out["results"] = results
    return out


def serp_overview_or_top(serp: Any, overview_chars: int = 1000, top_results: int = 3) -> Any:
    """The AI overview (first ``overview_chars`` characters) when the SERP has one;
    otherwise the top ``top_results`` organic results as title and snippet."""
    if not isinstance(serp, dict):
        return serp
    overview = serp.get("ai_overview")
    if isinstance(overview, dict):
        overview = overview.get("text")
    if isinstance(overview, str) and overview.strip():
        return {"ai_overview": overview.strip()[:overview_chars]}
    results = [
        {k: f[k] for k in ("title", "snippet") if f.get(k)}
        for f in serp.get("features") or []
        if isinstance(f, dict)
    ]
    return {"results": [r for r in results if r][:top_results]}


FORMATS = {"serp_compact": serp_compact, "serp_overview_or_top": serp_overview_or_top}


def build_state(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if "state" in spec:
        state = {}
        for name, source in spec["state"].items():
            if isinstance(source, dict):
                value = get_path(row, source["from"])
                fmt = source.get("format")
                options = {k: v for k, v in source.items() if k not in ("from", "format")}
                state[name] = FORMATS[fmt](value, **options) if fmt else value
            else:
                state[name] = get_path(row, source)
        return state
    return {f: row.get(f) for f in spec["state_fields"] if f != spec.get("variant_field")}


def canonical(row: dict[str, Any], task: str, spec: dict[str, Any]) -> dict[str, Any]:
    """One canonical record. Raises ValueError on a label the task cannot take."""
    raw_label = str(get_path(row, spec.get("label_field", "label")))
    label = spec.get("label_map", {}).get(raw_label, raw_label)
    if label not in spec["final_labels"]:
        raise ValueError(f"{row.get('id')}: label {label!r} not in {spec['final_labels']}")

    variant = None
    if spec.get("variant_field"):
        raw_variant = get_path(row, spec["variant_field"])
        variant_map = spec.get("variant_map")
        if variant_map:
            if raw_variant not in variant_map:
                raise ValueError(f"{row.get('id')}: {spec['variant_field']}={raw_variant!r} not in {list(variant_map)}")
            variant = variant_map[raw_variant]
        else:
            variant = "brand" if raw_variant in (True, "brand", "브랜드", 1, "1", "true") else "nonbrand"
        allowed = (spec.get("variant_labels") or {}).get(variant)
        if allowed is not None and label not in allowed:
            raise ValueError(f"{row.get('id')}: label {label!r} not allowed for {variant} (allowed {allowed})")

    state = build_state(row, spec)
    if variant is not None and spec.get("variant_in_state"):
        state[spec["variant_in_state"]] = variant == "brand"

    record = {
        "id": row["id"],
        "task": task,
        "lang": LANG_ALIASES.get(row["lang"], row["lang"]),
        "partition": row.get("partition") or row.get("part"),
        "state": state,
        "label": label,
        "variant": variant,
    }
    if spec.get("meta_fields"):
        record["meta"] = {name: get_path(row, path) for name, path in spec["meta_fields"].items()}
    return record


def convert(rows, task: str, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """All records, plus one error message per row that could not be converted."""
    records, errors = [], []
    seen: set[str] = set()
    for row in rows:
        try:
            record = canonical(row, task, spec)
        except (ValueError, KeyError) as error:
            errors.append(str(error))
            continue
        if record["id"] in seen:
            errors.append(f"{record['id']}: duplicate id")
            continue
        seen.add(record["id"])
        records.append(record)
    return records, errors
