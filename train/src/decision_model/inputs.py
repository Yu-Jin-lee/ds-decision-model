"""Model input: raw fields kept in the data, assembled into ``state`` at load time.

Canonical records store each input field raw, as it is in the task master
(``record["inputs"]``). The ``state`` the model sees is built when the data is
read, following the task's ``state`` spec in configs/tasks.yaml::

    state:
      keyword: keyword                       # copied as is
      serp: {from: serp, format: serp_overview_or_top, overview_chars: 1000}

So changing how an input is presented (a format or its options) is a config
change; the data in S3 does not need to be rebuilt or uploaded again.
"""

from __future__ import annotations

from typing import Any, Callable


def get_path(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _overview_text(serp: dict[str, Any]) -> str | None:
    overview = serp.get("ai_overview")
    if isinstance(overview, dict):
        overview = overview.get("text")
    return overview.strip() if isinstance(overview, str) and overview.strip() else None


def _results(serp: dict[str, Any]) -> list[dict[str, Any]]:
    results = [
        {k: f[k] for k in ("title", "snippet") if f.get(k)}
        for f in serp.get("features") or []
        if isinstance(f, dict)
    ]
    return [r for r in results if r]


def serp_compact(serp: Any) -> Any:
    """AI overview text and every result's title and snippet; drops ids and domains."""
    if not isinstance(serp, dict):
        return serp
    out: dict[str, Any] = {}
    overview = _overview_text(serp)
    if overview:
        out["ai_overview"] = overview
    results = _results(serp)
    if results:
        out["results"] = results
    return out


def serp_overview_or_top(serp: Any, overview_chars: int = 1000, top_results: int = 3) -> Any:
    """The AI overview (first ``overview_chars`` characters) when the SERP has one;
    otherwise the top ``top_results`` results as title and snippet."""
    if not isinstance(serp, dict):
        return serp
    overview = _overview_text(serp)
    if overview:
        return {"ai_overview": overview[:overview_chars]}
    return {"results": _results(serp)[:top_results]}


FORMATS: dict[str, Callable[..., Any]] = {
    "serp_compact": serp_compact,
    "serp_overview_or_top": serp_overview_or_top,
}


def _state_spec(spec: dict[str, Any]) -> dict[str, Any]:
    """``state`` as ``{name: source}``; ``state_fields`` (a list) is the short form."""
    if "state" in spec:
        return dict(spec["state"])
    return {f: f for f in spec["state_fields"] if f != spec.get("variant_field")}


def extract_inputs(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """Raw value of every state field, keyed by its state name (no formatting)."""
    inputs = {}
    for name, source in _state_spec(spec).items():
        path = source["from"] if isinstance(source, dict) else source
        inputs[name] = get_path(row, path)
    return inputs


def assemble_state(inputs: dict[str, Any], variant: str | None, spec: dict[str, Any]) -> dict[str, Any]:
    """The model's ``state``: raw inputs with each field's format applied."""
    state = {}
    for name, source in _state_spec(spec).items():
        value = inputs.get(name)
        if isinstance(source, dict) and source.get("format"):
            options = {k: v for k, v in source.items() if k not in ("from", "format")}
            value = FORMATS[source["format"]](value, **options)
        state[name] = value
    if variant is not None and spec.get("variant_in_state"):
        state[spec["variant_in_state"]] = variant == "brand"
    return state
